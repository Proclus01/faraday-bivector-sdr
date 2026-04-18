# faraday-bivector-sdr

Faraday-bivector-centric SDR framework in hexagonal architecture.

Streams are modeled as projections of the electromagnetic field with attached metadata such as operating mode, polarization basis, frame reference, gain, and processing lineage. The project is designed for simulation-first DSP development while remaining usable with real SDR hardware through SoapySDR.

## Current status

`faraday-bivector-sdr` is a usable research and prototyping framework.

It already supports simulation, live receive pipelines, storage, beamforming experiments, range-Doppler products, and a local web UI. Transmit support exists in the Soapy adapter and direct scripts, but is not yet fully integrated into the declarative YAML pipeline model.

## Capability matrix

| Area | Status | Notes |
|---|---|---|
| Simulation | Ready | Simulation adapter, tones, noise, delays, YAML examples, dynamic operators |
| Live RX | Ready | SoapySDR receive pipelines for bladeRF, HackRF, PlutoSDR, and Hermes |
| Live TX | Partial | Supported in direct scripts through the Soapy adapter; not yet first-class in YAML configs |
| Radar | Prototype | Dechirp, range-Doppler, HTML viewers, early FMCW and polarimetric workflows |
| Beamforming | Prototype | Narrowband beamforming, time/phase alignment, array simulation examples |
| Web UI | Prototype | Local browser UI with parameter sliders, device controls, and live taps |
| Storage | Ready | NPZ record/playback, Zarr recording, streaming Zarr writer |
| Deployment readiness | Prototype | Good for local development and experimentation; not yet production-hardened |

## Highlights

- Simulation-first SDR pipelines with YAML configuration
- Real hardware receive through SoapySDR
- Spectrum estimation, frequency shifting, FIR filtering, and decimation
- Dynamic runtime operators for live parameter control
- Narrowband beamforming and synchronization helpers
- Range-Doppler processing and polarimetric map generation
- NPZ and Zarr recording support
- Local web UI for controls and visualization

## Quick start

Install from source:

```bash
pip install -e .
```

Optional dependencies:
- `SoapySDR` for live hardware
- `numba` for accelerated FIR convolution
- `zarr` for Zarr-based recording

### Basic simulated receive

```bash
python -m faraday_bivector_sdr.cli.fbsdr examples/basic_rx.yaml
```

### Web UI demo

```bash
python -m faraday_bivector_sdr.cli.fbsdr examples/webui_sim.yaml
```

Then open:

```text
http://localhost:8765
```

### Polarimetric map demo

```bash
python -m faraday_bivector_sdr.cli.fbsdr examples/polmap_sim.yaml
```

Open `polmap.html` in your browser.

## Real hardware

Live receive is available through SoapySDR. Example configs are included for:

- bladeRF
- HackRF
- PlutoSDR
- Hermes

Example:

```bash
python -m faraday_bivector_sdr.cli.fbsdr examples/soapy_bladerf.yaml
```

You can also list visible Soapy devices with:

```bash
python -m faraday_bivector_sdr.cli.fbsdr devices
```

## Soapy on macOS

Install SoapySDR with Homebrew:

```bash
brew install soapysdr
```

Then expose the Python module inside your virtual environment:

```bash
source env/soapy_site.sh
```

## TX status

Transmit support is implemented in the Soapy adapter and used by direct scripts such as:

- `run_txrx_2425.py`
- `run_polmap_webui_2425.py`

At present:

- RX via YAML is supported
- TX via scripts is supported
- TX via YAML is still in progress

## Project structure

- `domain/`: core types and DSP operators
- `ports/`: abstract device, storage, and timing interfaces
- `adapters/`: simulation, Soapy, storage, HTML, and web UI adapters
- `application/`: config loading, orchestration, CLI wiring
- `runtime/`: taps, control registry, parameters, and local web server

## Scope

This project is currently best viewed as a cleanly structured SDR framework for:

- simulation-first DSP development
- live SDR receive experiments
- early beamforming and radar workflows
- metadata-rich processing pipelines

It is not yet a production-ready SDR platform.

## License

MIT