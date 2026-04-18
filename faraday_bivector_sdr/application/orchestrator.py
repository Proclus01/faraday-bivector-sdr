from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable
from ..domain.types import FaradayProjection
from ..domain.operators.frequency import frequency_shift
from ..domain.operators.filters import bandpass_fir, decimate, fractional_delay
from ..domain.operators.spectral import estimate_spectrum
from ..domain.operators.beamforming import beamform_narrowband, align_time_phase
from ..domain.operators.radar import range_doppler
from ..domain.operators.dechirp import dechirp_local
from ..domain.operators.tee import tee
from ..domain.operators.polarization_stream import pol_xy_to_rl, select
from ..domain.operators.dynamic import dyn_frequency_shift, dyn_gain
from ..domain.operators.polmap import polmap_from_rd
from ..adapters.devices.sim import SimDeviceAdapter
from ..adapters.devices.soapy import SoapyDeviceAdapter, SoapyNotAvailable
from ..adapters.devices.file_npz import NPZPlaybackAdapter
from .configs.loader import load_config
from .configs.schema import Config, DeviceDef
from ..ports.device import RadioDevicePort
from ..runtime.engine import (
    consume_spectrum_console,
    record_npz_projection,
    record_npz_spectrum,
    consume_spectrum_collect_html,
    consume_range_doppler_html,
    consume_polmap_html,
)
from ..runtime.webui import start_server
from ..runtime.control import GLOBAL_CONTROL
from ..runtime.stream import tee_async
from ..runtime.taps import feed_spectrum_tap, feed_rd_tap, feed_polmap_tap

DeviceFactory = Callable[[str], RadioDevicePort]

@dataclass
class Orchestrator:
    device_factories: Dict[str, DeviceFactory]

    def _create_device(self, dev_def: DeviceDef) -> RadioDevicePort:
        if dev_def.adapter not in self.device_factories:
            raise ValueError(f"Unknown adapter: {dev_def.adapter}")
        dev = self.device_factories[dev_def.adapter](dev_def.args)
        GLOBAL_CONTROL.register_device(dev_def.id, dev)
        return dev

    def build(self, cfg: Config) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        devices: Dict[str, RadioDevicePort] = {}
        for d in cfg.devices:
            try:
                devices[d.id] = self._create_device(d)
            except SoapyNotAvailable as e:
                raise RuntimeError(str(e) + " (Hint: source env/soapy_site.sh and ensure hardware is connected).")
            except Exception as e:
                if "Device::make() no match" in str(e):
                    import SoapySDR  # type: ignore
                    found = SoapySDR.Device.enumerate()  # type: ignore
                    raise RuntimeError(f"No Soapy device matched args='{d.args}'. Found={found}. Is the device connected/powered? Try: SoapySDRUtil --find") from e
                raise

        artifacts: Dict[str | Tuple[str, int], Any] = {}
        for d in cfg.devices:
            for ch_idx, ch_cfg in enumerate(d.rx_channels):
                artifacts[(d.id, ch_idx)] = devices[d.id].open_rx(ch_cfg)

        projections: Dict[str, Any] = {}
        for p in cfg.projections:
            key = (p.device, p.channel)
            if key not in artifacts:
                raise ValueError(f"Projection references unknown device/channel: {key}")
            projections[p.id] = artifacts[key]

        def resolve_input(name: str) -> Any:
            if name in artifacts:
                return artifacts[name]
            if name in projections:
                return projections[name]
            raise KeyError(f"Input '{name}' not found among pipeline artifacts or projections.")

        for pl in cfg.pipelines:
            if pl.inputs is not None:
                current: Any = [resolve_input(i) for i in pl.inputs]
            elif pl.input is not None:
                current = resolve_input(pl.input)
            else:
                raise ValueError(f"Pipeline {pl.id} missing input(s)")

            for op in pl.ops:
                (name, params), = op.items()
                if name == "frequency_shift":
                    current = frequency_shift(current, **params)
                elif name == "bandpass":
                    current = bandpass_fir(current, **params)
                elif name == "decimate":
                    current = decimate(current, **params)
                elif name == "spectrum":
                    current = estimate_spectrum(current, **params)
                elif name == "align_time_phase":
                    current = align_time_phase(current, **params)
                elif name == "beamform_narrowband":
                    current = beamform_narrowband(current, **params)
                elif name == "range_doppler":
                    current = range_doppler(current, **params)
                elif name == "tee":
                    current = tee(current, **params)
                elif name == "fractional_delay":
                    current = fractional_delay(current, **params)
                elif name == "dechirp_local":
                    current = dechirp_local(current, **params)
                elif name == "pol_xy_to_rl":
                    current = pol_xy_to_rl(current)
                elif name == "select":
                    current = select(current, **params)
                elif name == "dyn_frequency_shift":
                    current = dyn_frequency_shift(current, **params)
                elif name == "dyn_gain":
                    current = dyn_gain(current, **params)
                elif name == "polmap":
                    current = polmap_from_rd(current, **params)
                else:
                    raise ValueError(f"Unknown operator: {name}")
            artifacts[pl.id] = current
        return projections, artifacts

async def run_config(path: str) -> None:
    cfg = load_config(path)
    factories: Dict[str, DeviceFactory] = {
        "sim": lambda args: SimDeviceAdapter(args),
        "soapy": lambda args: SoapyDeviceAdapter(args),
        "file_npz": lambda args: NPZPlaybackAdapter(args),
    }
    orch = Orchestrator(device_factories=factories)
    projections, artifacts = orch.build(cfg)

    tasks: List[asyncio.Task] = []

    # WebUI: start server, wire taps and keep-alive if requested
    webui_cfg = None
    for out in cfg.outputs:
        if out.type == "webui":
            webui_cfg = out.params
            break
    if webui_cfg:
        port = int(webui_cfg.get("port", 8765))
        # taps config: [{"pipeline":"pipe_id","type":"spectrum","id":"spec1"}, ...]
        taps_cfg = webui_cfg.get("taps", [])
        # Build a minimal config for UI initialization (params, devices, taps)
        config = {
            "params": webui_cfg.get("params", []),
            "devices": webui_cfg.get("devices", []),
            "taps": taps_cfg,
        }
        start_server(port, config)
        print(f"[WebUI] Listening on http://localhost:{port}")
        # For each tap, tee the stream and feed an observer
        for tap in taps_cfg:
            pid = tap.get("pipeline"); typ = tap.get("type"); tap_id = tap.get("id")
            if not pid or pid not in artifacts:
                print(f"[WebUI] WARNING: tap pipeline not found: {pid}"); continue
            stream = artifacts[pid]
            # Split stream: one for other outputs (replace artifacts[pid]), one for tap consumer
            split = await tee_async(stream, copies=2)
            artifacts[pid] = split[0]
            if typ == "spectrum":
                tasks.append(asyncio.create_task(feed_spectrum_tap(split[1], tap_id)))
            elif typ == "rd":
                tasks.append(asyncio.create_task(feed_rd_tap(split[1], tap_id)))
            elif typ == "polmap":
                tasks.append(asyncio.create_task(feed_polmap_tap(split[1], tap_id)))
            else:
                print(f"[WebUI] WARNING: unknown tap type {typ} for {pid}")
        # Keep-alive if asked
        hold = bool(webui_cfg.get("hold", True))
        if hold:
            async def _hold_forever():
                await asyncio.Event().wait()
            tasks.append(asyncio.create_task(_hold_forever()))

    # Other outputs
    for out in cfg.outputs:
        if out.type == "spectrum_console":
            pl_id = out.params.get("pipeline"); frames = int(out.params.get("frames", 8)); topk = int(out.params.get("topk", 3))
            stream = artifacts.get(pl_id)
            if stream is None: raise ValueError(f"Output references unknown pipeline: {pl_id}")
            tasks.append(asyncio.create_task(consume_spectrum_console(stream, frames=frames, topk=topk)))
        elif out.type == "record_npz":
            src_id = out.params.get("source"); path_out = out.params.get("path", "capture.npz"); max_frames = out.params.get("max_frames", 32)
            src = artifacts.get(src_id, projections.get(src_id))
            if src is None: raise ValueError(f"record_npz source not found: {src_id}")
            if isinstance(src, FaradayProjection):
                tasks.append(asyncio.create_task(record_npz_projection(src, path_out, max_frames=max_frames)))
            else:
                tasks.append(asyncio.create_task(record_npz_spectrum(src, path_out, max_frames=max_frames)))
        elif out.type == "html_waterfall":
            pl_id = out.params.get("pipeline"); path_out = out.params.get("path", "waterfall.html"); frames = int(out.params.get("frames", 16))
            stream = artifacts.get(pl_id)
            if stream is None: raise ValueError(f"html_waterfall pipeline not found: {pl_id}")
            tasks.append(asyncio.create_task(consume_spectrum_collect_html(stream, path_out, frames=frames)))
        elif out.type == "html_rangedoppler":
            pl_id = out.params.get("pipeline"); path_out = out.params.get("path", "rangedoppler.html"); frames = int(out.params.get("frames", 1))
            stream = artifacts.get(pl_id)
            if stream is None: raise ValueError(f"html_rangedoppler pipeline not found: {pl_id}")
            tasks.append(asyncio.create_task(consume_range_doppler_html(stream, path_out, frames=frames)))
        elif out.type == "html_polmap":
            pl_id = out.params.get("pipeline"); path_out = out.params.get("path", "polmap.html"); frames = int(out.params.get("frames", 1))
            stream = artifacts.get(pl_id)
            if stream is None: raise ValueError(f"html_polmap pipeline not found: {pl_id}")
            tasks.append(asyncio.create_task(consume_polmap_html(stream, path_out, frames=frames)))
        elif out.type == "webui":
            pass
        else:
            raise ValueError(f"Unknown output type: {out.type}")

    if tasks:
        await asyncio.gather(*tasks)
