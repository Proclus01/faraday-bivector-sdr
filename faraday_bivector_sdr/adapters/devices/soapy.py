from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Optional, Set

import numpy as np

from ...domain.types import (
    BufferFrame,
    FaradayProjection,
    FrameRef,
    Mode,
    Pattern,
    PolarizationBasis,
    ProjectionMeta,
)
from ...ports.device import RadioDevicePort, TxHandle
from ...ports.timing import TimingSyncPort


class SoapyNotAvailable(Exception):
    pass


# Common SoapySDR return codes (negative values)
SOAPY_SDR_TIMEOUT = -1
SOAPY_SDR_STREAM_ERROR = -2
SOAPY_SDR_CORRUPTION = -3
SOAPY_SDR_OVERFLOW = -4
SOAPY_SDR_NOT_SUPPORTED = -5
SOAPY_SDR_TIME_ERROR = -6
SOAPY_SDR_UNDERFLOW = -7


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _soapy_result_ret(obj: Any) -> int:
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return _safe_int(getattr(obj, "ret", 0), 0)


def _soapy_result_time_ns(obj: Any) -> Optional[int]:
    if isinstance(obj, (int, np.integer)):
        return None
    try:
        t = getattr(obj, "timeNs", 0)
        if t:
            return int(t)
    except Exception:
        return None
    return None


def _bytes_per_elem_for_format(fmt_name: str) -> int:
    f = fmt_name.upper()
    if f == "CF32":
        return 8
    if f == "CS16":
        return 4
    return 8


@dataclass(frozen=True)
class _RxConfig:
    channel: int
    sample_rate_hz: float
    center_freq_hz: float
    bandwidth_hz: float
    gain_db: float
    antenna: Optional[str]
    block_size: int
    timeout_us: int = 100_000


@dataclass(frozen=True)
class _TxConfig:
    channel: int
    sample_rate_hz: float
    center_freq_hz: float
    bandwidth_hz: float
    gain_db: float
    antenna: Optional[str]
    timeout_us: int = 100_000


class _RxWorker:
    """
    RX worker thread calling readStream().

    Your SoapySDR bindings accept 3 or 4 args for readStream. We use an adaptive strategy:
      1) keyword timeoutUs=...
      2) positional timeoutUs as 4th arg
      3) no-timeout fallback

    Important behavior:
    - TIMEOUT (-1) and OVERFLOW (-4) are not fatal; keep reading.
    """

    def __init__(self, *, sdr: Any, stream: Any, cfg: _RxConfig) -> None:
        self._sdr = sdr
        self._stream = stream
        self._cfg = cfg

        self._stop_event = threading.Event()
        self._faulted = False

        self._reported_fatal = False
        self._reported_overflow = False

        self._queue: "queue.Queue[Optional[BufferFrame]]" = queue.Queue(maxsize=16)
        self._thread = threading.Thread(
            target=self._run,
            name=f"fbsdr-soapy-rx-ch{cfg.channel}",
            daemon=True,
        )

    @property
    def faulted(self) -> bool:
        return self._faulted

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout_s: float) -> bool:
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def _emit_end(self) -> None:
        try:
            self._queue.put_nowait(None)
            return
        except Exception:
            pass
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

    def _push_drop_oldest(self, fr: BufferFrame) -> None:
        try:
            self._queue.put_nowait(fr)
            return
        except queue.Full:
            pass
        try:
            _ = self._queue.get_nowait()
        except Exception:
            pass
        try:
            self._queue.put_nowait(fr)
        except Exception:
            pass

    def _read_stream(self, buff: np.ndarray) -> Any:
        # 1) keyword timeoutUs
        try:
            return self._sdr.readStream(
                self._stream,
                [buff],
                int(buff.shape[0]),
                timeoutUs=int(self._cfg.timeout_us),
            )
        except TypeError:
            pass

        # 2) positional timeoutUs (4th arg)
        try:
            return self._sdr.readStream(
                self._stream,
                [buff],
                int(buff.shape[0]),
                int(self._cfg.timeout_us),
            )
        except TypeError:
            pass

        # 3) no-timeout fallback
        return self._sdr.readStream(
            self._stream,
            [buff],
            int(buff.shape[0]),
        )

    def _run(self) -> None:
        buff = np.zeros(int(self._cfg.block_size), dtype=np.complex64)
        try:
            while not self._stop_event.is_set():
                try:
                    ret = self._read_stream(buff)
                except Exception as e:
                    self._faulted = True
                    if not self._reported_fatal:
                        self._reported_fatal = True
                        print(f"[Soapy RX] ch={self._cfg.channel} readStream exception: {type(e).__name__}: {e}")
                    break

                n = _soapy_result_ret(ret)

                # Non-fatal conditions: keep reading
                if n == SOAPY_SDR_TIMEOUT:
                    continue
                if n == SOAPY_SDR_OVERFLOW:
                    # Log once and continue. Overflow means dropped samples; still recoverable.
                    if not self._reported_overflow:
                        self._reported_overflow = True
                        print(f"[Soapy RX] ch={self._cfg.channel} overflow (-4): host fell behind; continuing...")
                    continue
                if n == SOAPY_SDR_TIME_ERROR:
                    # Also recoverable in many setups.
                    continue

                # Fatal / stop conditions
                if n < 0:
                    self._faulted = True
                    if not self._reported_fatal:
                        self._reported_fatal = True
                        print(f"[Soapy RX] ch={self._cfg.channel} readStream ret={n} (fatal)")
                    break

                if n == 0:
                    continue

                out = buff[:n].copy()
                ts = _soapy_result_time_ns(ret)
                self._push_drop_oldest(BufferFrame(samples=out, timestamp_ns=ts))

        finally:
            self._emit_end()

    async def frames(self) -> AsyncIterator[BufferFrame]:
        while True:
            item = await asyncio.to_thread(self._queue.get)
            if item is None:
                return
            yield item


class _TxWorker:
    """
    TX worker thread.

    Uses buflen-derived chunk sizing for bladeRF stability.
    Uses adaptive writeStream signature:
      1) keyword timeoutUs
      2) positional timeoutUs as 4th arg
      3) no-timeout fallback

    When disabled, it idles (no writeStream calls).
    """

    def __init__(
        self,
        *,
        sdr: Any,
        stream: Any,
        cfg: _TxConfig,
        waveform: np.ndarray,
        repeat: bool,
        stream_format: str,
        buflen_bytes: int,
        initial_enabled: bool,
    ) -> None:
        self._sdr = sdr
        self._stream = stream
        self._cfg = cfg
        self._repeat = bool(repeat)

        self._enabled_lock = threading.Lock()
        self._enabled = bool(initial_enabled)

        self._stop_event = threading.Event()
        self._faulted = False

        bpe = _bytes_per_elem_for_format(stream_format)
        if buflen_bytes > 0 and (buflen_bytes % bpe) == 0:
            chunk_elems = int(buflen_bytes // bpe)
        else:
            chunk_elems = 1024
        self._chunk_elems = max(1, int(chunk_elems))

        w = waveform.astype(np.complex64, copy=False)
        n = int(w.shape[0])
        c = int(self._chunk_elems)
        chunks = (n + c - 1) // c
        padded_len = chunks * c

        padded = np.zeros(padded_len, dtype=np.complex64)
        padded[:n] = w
        self._chunks = padded.reshape(chunks, c)

        self._thread = threading.Thread(
            target=self._run,
            name=f"fbsdr-soapy-tx-ch{cfg.channel}",
            daemon=True,
        )

    @property
    def faulted(self) -> bool:
        return self._faulted

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout_s: float) -> bool:
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def set_enabled(self, enabled: bool) -> None:
        with self._enabled_lock:
            self._enabled = bool(enabled)

    def _is_enabled(self) -> bool:
        with self._enabled_lock:
            return bool(self._enabled)

    def _write_stream(self, buf: np.ndarray) -> Any:
        # 1) keyword timeoutUs
        try:
            return self._sdr.writeStream(
                self._stream,
                [buf],
                int(buf.shape[0]),
                timeoutUs=int(self._cfg.timeout_us),
            )
        except TypeError:
            pass

        # 2) positional timeoutUs as 4th arg
        try:
            return self._sdr.writeStream(
                self._stream,
                [buf],
                int(buf.shape[0]),
                int(self._cfg.timeout_us),
            )
        except TypeError:
            pass

        # 3) no-timeout fallback
        return self._sdr.writeStream(
            self._stream,
            [buf],
            int(buf.shape[0]),
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if not self._is_enabled():
                    time.sleep(0.002)
                    continue

                for i in range(self._chunks.shape[0]):
                    if self._stop_event.is_set() or (not self._is_enabled()):
                        break

                    chunk = self._chunks[i]
                    try:
                        ret = self._write_stream(chunk)
                    except Exception:
                        self._faulted = True
                        return

                    n = _soapy_result_ret(ret)

                    # Non-fatal: underflow (TX starved) can be recoverable
                    if n == SOAPY_SDR_UNDERFLOW:
                        continue
                    if n == SOAPY_SDR_TIMEOUT:
                        continue

                    if n < 0:
                        self._faulted = True
                        return

                if not self._repeat:
                    return
        finally:
            pass


class SoapyTxHandle(TxHandle):
    """
    TxHandle with persisted enable state; may be toggled before start().
    """

    def __init__(
        self,
        *,
        start_cb: Callable[[], None],
        stop_cb: Callable[[], None],
        set_enabled_cb: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._start_cb = start_cb
        self._stop_cb = stop_cb
        self._set_enabled_cb = set_enabled_cb

        self._started = False
        self._stopping = False
        self._desired_enabled = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._start_cb()
        self.set_enabled(self._desired_enabled)

    async def stop_async(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._stop_cb()
        await asyncio.sleep(0)

    def stop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._stop_cb()
            return
        loop.create_task(self.stop_async())

    def set_enabled(self, enabled: bool) -> None:
        self._desired_enabled = bool(enabled)
        if self._set_enabled_cb is None:
            return
        try:
            self._set_enabled_cb(bool(enabled))
        except Exception:
            pass


class SoapyDeviceAdapter(RadioDevicePort, TimingSyncPort):
    def __init__(self, args: str = "") -> None:
        try:
            import SoapySDR  # type: ignore
        except Exception as e:
            raise SoapyNotAvailable(
                "SoapySDR Python bindings not available (use Homebrew and source env/soapy_site.sh)"
            ) from e

        self._SoapySDR = SoapySDR
        self._args = args
        self._sdr = SoapySDR.Device(args)  # type: ignore

        self._lock = threading.RLock()
        self._closed = False
        self._closed_stream_ids: Set[int] = set()

        self._is_bladerf = ("bladerf" in args.lower())
        self._skip_close_stream = self._is_bladerf

        self._rx_format_name = "CF32"
        self._tx_format_name = "CF32"
        self._rx_format = self._SoapySDR.SOAPY_SDR_CF32
        self._tx_format = self._SoapySDR.SOAPY_SDR_CF32

        self._bladerf_stream_args: Dict[str, str] = {
            "buffers": "32",
            "buflen": "32768",  # multiple of 1024; 32768/8=4096 CF32 elems
            "transfers": "0",
        }
        self._bladerf_buflen_bytes = int(self._bladerf_stream_args["buflen"])

        self._rx_streams: Dict[int, Any] = {}
        self._rx_cfg: Dict[int, _RxConfig] = {}
        self._rx_workers: Dict[int, _RxWorker] = {}
        self._rx_started: Set[int] = set()

        self._tx_streams: Dict[int, Any] = {}
        self._tx_cfg: Dict[int, _TxConfig] = {}
        self._tx_workers: Dict[int, _TxWorker] = {}
        self._tx_started: Set[int] = set()

    def _mark_stream_closed(self, st: Any) -> bool:
        with self._lock:
            sid = id(st)
            if sid in self._closed_stream_ids:
                return False
            self._closed_stream_ids.add(sid)
            return True

    def _activate_stream_best_effort(self, st: Any) -> None:
        try:
            self._sdr.activateStream(st)
        except Exception:
            pass

    def _deactivate_stream_best_effort(self, st: Any) -> None:
        try:
            self._sdr.deactivateStream(st)
        except Exception:
            pass

    def _close_stream_best_effort(self, st: Any) -> None:
        if not self._mark_stream_closed(st):
            return
        self._deactivate_stream_best_effort(st)
        if self._skip_close_stream:
            return
        time.sleep(0.03)
        try:
            self._sdr.closeStream(st)
        except Exception:
            pass

    def _ensure_rx_started(self, ch: int) -> None:
        with self._lock:
            if ch in self._rx_started:
                return
            st = self._rx_streams[ch]
            cfg = self._rx_cfg[ch]
            worker = _RxWorker(sdr=self._sdr, stream=st, cfg=cfg)
            self._rx_workers[ch] = worker
            self._rx_started.add(ch)
        self._activate_stream_best_effort(st)
        worker.start()

    def _ensure_tx_started(self, ch: int, waveform: np.ndarray, repeat: bool) -> _TxWorker:
        with self._lock:
            if ch in self._tx_started:
                return self._tx_workers[ch]
            st = self._tx_streams[ch]
            cfg = self._tx_cfg[ch]
            worker = _TxWorker(
                sdr=self._sdr,
                stream=st,
                cfg=cfg,
                waveform=waveform,
                repeat=repeat,
                stream_format=self._tx_format_name,
                buflen_bytes=(self._bladerf_buflen_bytes if self._is_bladerf else 0),
                initial_enabled=False,
            )
            self._tx_workers[ch] = worker
            self._tx_started.add(ch)
        self._activate_stream_best_effort(st)
        worker.start()
        return worker

    def _stop_tx_channel(self, ch: int, *, join_timeout_s: float = 5.0) -> None:
        with self._lock:
            st = self._tx_streams.get(ch)
            worker = self._tx_workers.get(ch)
        if worker is not None:
            worker.request_stop()
        if st is not None:
            self._deactivate_stream_best_effort(st)
        if worker is not None:
            joined = worker.join(join_timeout_s)
            if not joined:
                return
            if worker.faulted and st is not None:
                self._mark_stream_closed(st)
                return
        if st is not None:
            self._close_stream_best_effort(st)
        with self._lock:
            self._tx_streams.pop(ch, None)
            self._tx_workers.pop(ch, None)
            self._tx_cfg.pop(ch, None)
            self._tx_started.discard(ch)

    def _stop_rx_channel(self, ch: int, *, join_timeout_s: float = 5.0) -> None:
        with self._lock:
            st = self._rx_streams.get(ch)
            worker = self._rx_workers.get(ch)
        if worker is not None:
            worker.request_stop()
        if st is not None:
            self._deactivate_stream_best_effort(st)
        if worker is not None:
            joined = worker.join(join_timeout_s)
            if not joined:
                return
            if worker.faulted and st is not None:
                self._mark_stream_closed(st)
                return
        if st is not None:
            self._close_stream_best_effort(st)
        with self._lock:
            self._rx_streams.pop(ch, None)
            self._rx_workers.pop(ch, None)
            self._rx_cfg.pop(ch, None)
            self._rx_started.discard(ch)

    def open_rx(self, chan_cfg: Dict[str, Any]) -> FaradayProjection:
        SoapySDR = self._SoapySDR
        SOAPY_SDR_RX = SoapySDR.SOAPY_SDR_RX

        ch = int(chan_cfg.get("channel", 0))
        sr = float(chan_cfg["sample_rate_hz"])
        cf = float(chan_cfg["center_freq_hz"])
        bw = float(chan_cfg.get("bandwidth_hz", sr))
        gain = float(chan_cfg.get("gain_db", 0.0))
        ant = chan_cfg.get("antenna", None)
        block_size = int(chan_cfg.get("block_size", 8192))

        cfg = _RxConfig(
            channel=ch,
            sample_rate_hz=sr,
            center_freq_hz=cf,
            bandwidth_hz=bw,
            gain_db=gain,
            antenna=ant,
            block_size=block_size,
        )

        self._sdr.setSampleRate(SOAPY_SDR_RX, ch, sr)
        self._sdr.setFrequency(SOAPY_SDR_RX, ch, cf)

        try:
            self._sdr.setBandwidth(SOAPY_SDR_RX, ch, bw)
        except Exception:
            pass

        try:
            if ant is not None:
                self._sdr.setAntenna(SOAPY_SDR_RX, ch, ant)
        except Exception:
            pass

        try:
            self._sdr.setGain(SOAPY_SDR_RX, ch, gain)
        except Exception:
            pass

        try:
            st = self._sdr.setupStream(
                SOAPY_SDR_RX,
                self._rx_format,
                [ch],
                self._bladerf_stream_args if self._is_bladerf else {},
            )
        except TypeError:
            st = self._sdr.setupStream(SOAPY_SDR_RX, self._rx_format, [ch])  # type: ignore

        with self._lock:
            self._rx_streams[ch] = st
            self._rx_cfg[ch] = cfg

        async def gen() -> AsyncIterator[BufferFrame]:
            self._ensure_rx_started(ch)
            worker = self._rx_workers[ch]
            async for fr in worker.frames():
                yield fr

        meta = ProjectionMeta(
            mode=Mode(center_freq_hz=cf, sample_rate_hz=sr, bandwidth_hz=bw, lo_chain=(cf,)),
            polarization=PolarizationBasis(name="UNKNOWN", matrix_to_xy=np.eye(2, dtype=np.complex64)),
            pattern=Pattern("unknown"),
            frame=FrameRef("DEVICE", t0_ns=0),
            gain_db=gain,
            noise_figure_db=None,
            tags={"adapter": "soapy", "args": self._args, "channel": ch, "block_size": cfg.block_size},
        )
        return FaradayProjection(meta=meta, stream=gen())

    def open_tx(self, chan_cfg: Dict[str, Any], buffer: np.ndarray, repeat: bool = True) -> TxHandle:
        SoapySDR = self._SoapySDR
        SOAPY_SDR_TX = SoapySDR.SOAPY_SDR_TX

        ch = int(chan_cfg.get("channel", 0))
        sr = float(chan_cfg["sample_rate_hz"])
        cf = float(chan_cfg["center_freq_hz"])
        bw = float(chan_cfg.get("bandwidth_hz", sr))
        gain = float(chan_cfg.get("gain_db", 0.0))
        ant = chan_cfg.get("antenna", None)

        cfg = _TxConfig(
            channel=ch,
            sample_rate_hz=sr,
            center_freq_hz=cf,
            bandwidth_hz=bw,
            gain_db=gain,
            antenna=ant,
        )

        self._sdr.setSampleRate(SOAPY_SDR_TX, ch, sr)
        self._sdr.setFrequency(SOAPY_SDR_TX, ch, cf)

        try:
            self._sdr.setBandwidth(SOAPY_SDR_TX, ch, bw)
        except Exception:
            pass

        try:
            if ant is not None:
                self._sdr.setAntenna(SOAPY_SDR_TX, ch, ant)
        except Exception:
            pass

        try:
            self._sdr.setGain(SOAPY_SDR_TX, ch, gain)
        except Exception:
            pass

        try:
            st = self._sdr.setupStream(
                SOAPY_SDR_TX,
                self._tx_format,
                [ch],
                self._bladerf_stream_args if self._is_bladerf else {},
            )
        except TypeError:
            st = self._sdr.setupStream(SOAPY_SDR_TX, self._tx_format, [ch])  # type: ignore

        with self._lock:
            self._tx_streams[ch] = st
            self._tx_cfg[ch] = cfg

        worker_ref: Dict[str, _TxWorker] = {}

        def _start() -> None:
            worker_ref["w"] = self._ensure_tx_started(ch, waveform=buffer, repeat=repeat)

        def _stop() -> None:
            self._stop_tx_channel(ch)

        def _set_enabled(enabled: bool) -> None:
            w = worker_ref.get("w")
            if w is None:
                return
            w.set_enabled(enabled)

        return SoapyTxHandle(start_cb=_start, stop_cb=_stop, set_enabled_cb=_set_enabled)

    def set_clock_source(self, source: str) -> None:
        try:
            self._sdr.setClockSource(str(source))
        except Exception:
            pass

    def set_time_now(self, time_ns: int) -> None:
        try:
            self._sdr.setHardwareTime(int(time_ns), "TRIGGERED")
        except Exception:
            try:
                self._sdr.setHardwareTime(int(time_ns))
            except Exception:
                pass

    def get_time_now(self) -> int:
        try:
            return int(self._sdr.getHardwareTime(""))
        except Exception:
            return 0

    # Optional setters for control registry
    def set_rx_gain(self, channel: int, gain_db: float) -> None:  # pragma: no cover
        try:
            self._sdr.setGain(self._SoapySDR.SOAPY_SDR_RX, int(channel), float(gain_db))
        except Exception:
            pass

    def set_rx_center_freq(self, channel: int, cf_hz: float) -> None:  # pragma: no cover
        try:
            self._sdr.setFrequency(self._SoapySDR.SOAPY_SDR_RX, int(channel), float(cf_hz))
        except Exception:
            pass

    def set_rx_bandwidth(self, channel: int, bw_hz: float) -> None:  # pragma: no cover
        try:
            self._sdr.setBandwidth(self._SoapySDR.SOAPY_SDR_RX, int(channel), float(bw_hz))
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tx_channels = sorted(self._tx_streams.keys())
            rx_channels = sorted(self._rx_streams.keys())

        for ch in tx_channels:
            try:
                self._stop_tx_channel(int(ch))
            except Exception:
                pass

        for ch in rx_channels:
            try:
                self._stop_rx_channel(int(ch))
            except Exception:
                pass
