import asyncio
from faraday_bivector_sdr.application.orchestrator import Orchestrator
from faraday_bivector_sdr.application.configs.schema import Config, DeviceDef, ProjectionDef, PipelineDef, OutputDef
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter

def test_pipeline_to_pipeline_refs_builds():
    # Minimal config: one device channel -> RD -> alias pipeline
    cfg = Config(
        devices=[DeviceDef(id="sim", adapter="sim", args="", rx_channels=[{
            "center_freq_hz": 1e8, "sample_rate_hz": 1e6, "bandwidth_hz": 1e6, "block_size": 1024,
            "tone_offsets_hz": [100e3], "amplitudes": [1.0], "noise_std": 0.0, "seed": 1, "max_frames": 8
        }])],
        projections=[ProjectionDef(id="P0", device="sim", channel=0)],
        pipelines=[
            PipelineDef(id="RD0", input="P0", ops=[{"range_doppler": {"n_fast": 1024, "n_slow": 8}}]),
            PipelineDef(id="ALIAS", inputs=["RD0"], ops=[])  # consume pipeline output by name
        ],
        outputs=[]
    )
    orch = Orchestrator(device_factories={"sim": lambda args: SimDeviceAdapter(args)})
    projections, artifacts = orch.build(cfg)
    assert "RD0" in artifacts and "ALIAS" in artifacts
