from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional, Tuple

import numpy as np

from faraday_bivector_sdr.adapters.devices.soapy import SoapyDeviceAdapter, SoapyTxHandle
from faraday_bivector_sdr.domain.operators.dechirp import dechirp_local
from faraday_bivector_sdr.domain.operators.polmap import polmap_from_rd
from faraday_bivector_sdr.domain.operators.radar import range_doppler
from faraday_bivector_sdr.domain.operators.spectral import estimate_spectrum
from faraday_bivector_sdr.domain.operators.tee import tee
from faraday_bivector_sdr.domain.types import BufferFrame, FaradayProjection, RangeDopplerFrame, SpectralFrame
from faraday_bivector_sdr.runtime.control import GLOBAL_CONTROL
from faraday_bivector_sdr.runtime.stream import tee_async
from faraday_bivector_sdr.runtime.taps import feed_polmap_tap, feed_rd_tap, feed_spectrum_tap
from faraday_bivector_sdr.runtime.webui import start_server


@dataclass(frozen=True)
class RadarConfig:
    center_freq_hz: float = 2_425_000_000.0
    sample_rate_hz: float = 2_000_000.0
    bandwidth_hz: float = 2_000_000.0

    # Match bladeRF CF32 buflen=32768 -> 4096 elems/chunk
    n_fast: int = 4096
    n_slow: int = 32

    sweep_bw_hz: float = 800_000.0
    tdd_chirps: int = 16

    rx_gain_db: float = 25.0
    tx_gain_db: float = -23.75

    webui_port: int = 8765


def make_fmcw_chirp(cfg: RadarConfig) -> np.ndarray:
    sr = float(cfg.sample_rate_hz)
    n = int(cfg.n_fast)
    sweep_time_s = float(n) / sr
    k = float(cfg.sweep_bw_hz) / sweep_time_s

    t = np.arange(n, dtype=np.float64) / sr
    phase = 2.0 * np.pi * (0.5 * k * (t**2))
    chirp = np.exp(1j * phase).astype(np.complex64)
    chirp *= np.complex64(0.02 + 0j)
    return chirp


def _q_put_drop_oldest(q: "asyncio.Queue[Optional[BufferFrame]]", item: BufferFrame) -> None:
    try:
        q.put_nowait(item)
        return
    except asyncio.QueueFull:
        pass
    try:
        _ = q.get_nowait()
    except Exception:
        pass
    try:
        q.put_nowait(item)
    except Exception:
        pass


async def zip_by_arrival(
    a: AsyncIterator[BufferFrame], b: AsyncIterator[BufferFrame]
) -> AsyncIterator[Tuple[BufferFrame, BufferFrame]]:
    ita = a.__aiter__()
    itb = b.__aiter__()
    while True:
        try:
            fa = await ita.__anext__()
            fb = await itb.__anext__()
        except StopAsyncIteration:
            return
        yield fa, fb


def tdm_demux_hv(
    rx_h: FaradayProjection,
    rx_v: FaradayProjection,
    tx_h: SoapyTxHandle,
    tx_v: SoapyTxHandle,
    *,
    tdd_chirps: int,
) -> Tuple[FaradayProjection, FaradayProjection, FaradayProjection, FaradayProjection, asyncio.Task]:
    q_hh: "asyncio.Queue[Optional[BufferFrame]]" = asyncio.Queue(maxsize=16)
    q_hv: "asyncio.Queue[Optional[BufferFrame]]" = asyncio.Queue(maxsize=16)
    q_vh: "asyncio.Queue[Optional[BufferFrame]]" = asyncio.Queue(maxsize=16)
    q_vv: "asyncio.Queue[Optional[BufferFrame]]" = asyncio.Queue(maxsize=16)

    async def pump() -> None:
        chirp_idx = 0
        current_state = 0

        tx_h.set_enabled(True)
        tx_v.set_enabled(False)

        try:
            async for fr_h, fr_v in zip_by_arrival(rx_h.stream, rx_v.stream):
                state = (chirp_idx // max(1, int(tdd_chirps))) % 2
                if state != current_state:
                    current_state = state
                    if current_state == 0:
                        tx_h.set_enabled(True)
                        tx_v.set_enabled(False)
                    else:
                        tx_h.set_enabled(False)
                        tx_v.set_enabled(True)

                if current_state == 0:
                    _q_put_drop_oldest(q_hh, fr_h)
                    _q_put_drop_oldest(q_hv, fr_v)
                else:
                    _q_put_drop_oldest(q_vh, fr_h)
                    _q_put_drop_oldest(q_vv, fr_v)

                chirp_idx += 1

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[TDM] pump error: {type(e).__name__}: {e}")
        finally:
            try:
                tx_h.set_enabled(False)
                tx_v.set_enabled(False)
            except Exception:
                pass
            for q in (q_hh, q_hv, q_vh, q_vv):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

    task = asyncio.create_task(pump())

    async def stream_from_queue(q: "asyncio.Queue[Optional[BufferFrame]]") -> AsyncIterator[BufferFrame]:
        while True:
            item = await q.get()
            if item is None:
                return
            yield item

    hh = FaradayProjection(meta=rx_h.meta, stream=stream_from_queue(q_hh))
    hv = FaradayProjection(meta=rx_v.meta, stream=stream_from_queue(q_hv))
    vh = FaradayProjection(meta=rx_h.meta, stream=stream_from_queue(q_vh))
    vv = FaradayProjection(meta=rx_v.meta, stream=stream_from_queue(q_vv))
    return hh, hv, vh, vv, task


async def strip_rd_timestamps(stream: AsyncIterator[RangeDopplerFrame]) -> AsyncIterator[RangeDopplerFrame]:
    """
    Force timestamp_ns=None so polmap_from_rd uses arrival-order alignment.
    This avoids stalls when hardware timestamps across TDM streams drift.
    """
    async for fr in stream:
        yield RangeDopplerFrame(
            range_m=fr.range_m,
            doppler_hz=fr.doppler_hz,
            power_db=fr.power_db,
            timestamp_ns=None,
            meta=fr.meta,
        )


async def debug_spectrum(stream: AsyncIterator[SpectralFrame]) -> None:
    i = 0
    async for fr in stream:
        peak = float(np.max(fr.psd_db)) if fr.psd_db.size else float("nan")
        print(f"[DBG] spectrum frame={i} peak_db={peak:.1f}")
        i += 1


async def debug_polmap(stream: AsyncIterator[Any]) -> None:
    i = 0
    async for fr in stream:
        try:
            names = getattr(fr, "metric_names", [])
            print(f"[DBG] polmap frame={i} metrics={names}")
        except Exception:
            print(f"[DBG] polmap frame={i}")
        i += 1


async def main() -> None:
    cfg = RadarConfig()
    sweep_time_s = float(cfg.n_fast) / float(cfg.sample_rate_hz)
    pri_s = sweep_time_s

    ui_config = {
        "params": [],
        "devices": [
            {
                "id": "brf",
                "adapter": "soapy",
                "controls": [
                    {"field": "gain", "channel": 0, "min": 0, "max": 60, "step": 1, "init": cfg.rx_gain_db},
                    {"field": "gain", "channel": 1, "min": 0, "max": 60, "step": 1, "init": cfg.rx_gain_db},
                ],
            }
        ],
        "taps": [
            {"pipeline": "LIVE", "type": "spectrum", "id": "RXH"},
            {"pipeline": "LIVE", "type": "rd", "id": "HH"},
            {"pipeline": "LIVE", "type": "polmap", "id": "main"},
        ],
    }
    start_server(cfg.webui_port, ui_config)
    print(f"[WebUI] Listening on http://localhost:{cfg.webui_port}")

    stop_event = asyncio.Event()
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    except Exception:
        pass

    dev = SoapyDeviceAdapter("driver=bladerf")
    GLOBAL_CONTROL.register_device("brf", dev)

    tx_chirp = make_fmcw_chirp(cfg)
    tx_h: Optional[SoapyTxHandle] = None
    tx_v: Optional[SoapyTxHandle] = None

    demux_task: Optional[asyncio.Task] = None
    tasks: list[asyncio.Task] = []

    try:
        # RX0=V, RX1=H
        rx_v = dev.open_rx(
            dict(
                channel=0,
                center_freq_hz=cfg.center_freq_hz,
                sample_rate_hz=cfg.sample_rate_hz,
                bandwidth_hz=cfg.bandwidth_hz,
                gain_db=cfg.rx_gain_db,
                block_size=cfg.n_fast,
            )
        )
        rx_h = dev.open_rx(
            dict(
                channel=1,
                center_freq_hz=cfg.center_freq_hz,
                sample_rate_hz=cfg.sample_rate_hz,
                bandwidth_hz=cfg.bandwidth_hz,
                gain_db=cfg.rx_gain_db,
                block_size=cfg.n_fast,
            )
        )

        # Tee RX_H for spectrum + demux
        rx_h_spec, rx_h_demux = tee(rx_h, copies=2)

        # TX0=H, TX1=V
        tx_h = dev.open_tx(
            dict(
                channel=0,
                center_freq_hz=cfg.center_freq_hz,
                sample_rate_hz=cfg.sample_rate_hz,
                bandwidth_hz=cfg.bandwidth_hz,
                gain_db=cfg.tx_gain_db,
            ),
            tx_chirp,
            repeat=True,
        )
        tx_v = dev.open_tx(
            dict(
                channel=1,
                center_freq_hz=cfg.center_freq_hz,
                sample_rate_hz=cfg.sample_rate_hz,
                bandwidth_hz=cfg.bandwidth_hz,
                gain_db=cfg.tx_gain_db,
            ),
            tx_chirp,
            repeat=True,
        )

        tx_h.set_enabled(True)
        tx_v.set_enabled(False)
        tx_h.start()
        tx_v.start()

        # Spectrum -> tee: tap + debug
        spec_stream = estimate_spectrum(rx_h_spec, fft_size=2048, avg=4)
        spec_for_tap, spec_for_dbg = await tee_async(spec_stream, copies=2)
        tasks.append(asyncio.create_task(feed_spectrum_tap(spec_for_tap, "RXH")))
        tasks.append(asyncio.create_task(debug_spectrum(spec_for_dbg)))

        # Demux into HH/HV/VH/VV
        HH, HV, VH, VV, demux_task = tdm_demux_hv(
            rx_h=rx_h_demux,
            rx_v=rx_v,
            tx_h=tx_h,
            tx_v=tx_v,
            tdd_chirps=cfg.tdd_chirps,
        )

        # Dechirp
        HH_d = dechirp_local(HH, sweep_bw_hz=cfg.sweep_bw_hz, sweep_time_s=sweep_time_s)
        HV_d = dechirp_local(HV, sweep_bw_hz=cfg.sweep_bw_hz, sweep_time_s=sweep_time_s)
        VH_d = dechirp_local(VH, sweep_bw_hz=cfg.sweep_bw_hz, sweep_time_s=sweep_time_s)
        VV_d = dechirp_local(VV, sweep_bw_hz=cfg.sweep_bw_hz, sweep_time_s=sweep_time_s)

        # RD
        RD_HH_raw = range_doppler(HH_d, n_fast=cfg.n_fast, n_slow=cfg.n_slow, window_fast="hann", window_slow="hann", mode="pulsed")
        RD_HV_raw = range_doppler(HV_d, n_fast=cfg.n_fast, n_slow=cfg.n_slow, window_fast="hann", window_slow="hann", mode="pulsed")
        RD_VH_raw = range_doppler(VH_d, n_fast=cfg.n_fast, n_slow=cfg.n_slow, window_fast="hann", window_slow="hann", mode="pulsed")
        RD_VV_raw = range_doppler(VV_d, n_fast=cfg.n_fast, n_slow=cfg.n_slow, window_fast="hann", window_slow="hann", mode="pulsed")

        # Strip timestamps to force arrival-order alignment in polmap combiner
        RD_HH = strip_rd_timestamps(RD_HH_raw)
        RD_HV = strip_rd_timestamps(RD_HV_raw)
        RD_VH = strip_rd_timestamps(RD_VH_raw)
        RD_VV = strip_rd_timestamps(RD_VV_raw)

        # Tee HH RD for tap + polmap
        rd_hh_pol, rd_hh_tap = await tee_async(RD_HH, copies=2)
        tasks.append(asyncio.create_task(feed_rd_tap(rd_hh_tap, "HH")))

        # PolMap
        tolerance_pri = max(32, int(4 * cfg.tdd_chirps))
        POL_raw = polmap_from_rd(
            [rd_hh_pol, RD_HV, RD_VH, RD_VV],
            metrics=["span", "copol_ratio", "xpol_ratio"],
            pri_s=pri_s,
            tolerance_pri=tolerance_pri,
        )

        pol_for_tap, pol_for_dbg = await tee_async(POL_raw, copies=2)
        tasks.append(asyncio.create_task(feed_polmap_tap(pol_for_tap, "main")))
        tasks.append(asyncio.create_task(debug_polmap(pol_for_dbg)))

        await stop_event.wait()

    finally:
        for t in tasks:
            t.cancel()
        if demux_task is not None:
            demux_task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if demux_task is not None:
            await asyncio.gather(demux_task, return_exceptions=True)

        try:
            if tx_h is not None:
                tx_h.set_enabled(False)
            if tx_v is not None:
                tx_v.set_enabled(False)
        except Exception:
            pass

        if tx_h is not None:
            try:
                await tx_h.stop_async()
            except Exception:
                pass
        if tx_v is not None:
            try:
                await tx_v.stop_async()
            except Exception:
                pass

        try:
            dev.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
