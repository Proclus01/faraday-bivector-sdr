# faraday-bivector-sdr Roadmap

Patch 1:
- Core domain types and operators (frequency shift, FIR bandpass, decimate, spectrum).
- Simulation adapter.
- Orchestrator and CLI with YAML config.
- Minimal runtime consumer for spectrum.
- Tests and example config.

Patch 2:
- Device adapters: Soapy-based (BladeRF, HackRF, Pluto, Hermes).
- Timing and synchronization scaffolding (PPS, 10 MHz).
- Recording/Playback ports (NPZ native; Zarr optional).
- Polarization transforms and calibration stubs.

Patch 3 (this patch):
- Beamforming (narrowband), MIMO phase alignment, array calibration stubs.
- Radar imaging: streaming Range-Doppler, offline RD and simple 2D backprojection.
- HTML visualization for waterfall and range-Doppler.

Patch 4 (future):
- Wideband fractional-delay beamforming, MVDR/LCMV.
- Full MIMO synchronization with timestamp alignment and LO phase calibration.
- SAR processing pipelines and richer web UI.
