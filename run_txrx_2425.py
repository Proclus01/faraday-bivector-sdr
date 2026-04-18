import asyncio
from typing import Optional

import numpy as np

from faraday_bivector_sdr.adapters.devices.soapy import SoapyDeviceAdapter, SoapyTxHandle
from faraday_bivector_sdr.domain.operators.spectral import estimate_spectrum
from faraday_bivector_sdr.runtime.engine import consume_spectrum_console

SR = 2_000_000.0
CF = 2_425_000_000.0
BW = 2_000_000.0


def tone_buffer(freq_hz: float, amp: float = 0.01, n: int = 262144, sr: float = SR) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    x = amp * np.exp(1j * 2 * np.pi * (freq_hz / sr) * t)
    return x.astype(np.complex64)


async def main() -> None:
    dev = SoapyDeviceAdapter("driver=bladerf")
    tx0: Optional[SoapyTxHandle] = None

    try:
        # Configure RX but note: streams are activated lazily on first consumption.
        rx_v = dev.open_rx(
            dict(
                channel=0,
                center_freq_hz=CF,
                sample_rate_hz=SR,
                bandwidth_hz=BW,
                gain_db=0,
                block_size=8192,
            )
        )
        rx_h = dev.open_rx(
            dict(
                channel=1,
                center_freq_hz=CF,
                sample_rate_hz=SR,
                bandwidth_hz=BW,
                gain_db=0,
                block_size=8192,
            )
        )

        # Configure TX; activated on start().
        tx0_buf = tone_buffer(100_000.0, amp=0.01)
        tx0 = dev.open_tx(
            dict(
                channel=0,
                center_freq_hz=CF,
                sample_rate_hz=SR,
                bandwidth_hz=BW,
                gain_db=-30,
            ),
            tx0_buf,
            repeat=True,
        )
        tx0.start()

        # Observe spectra
        spec_v = estimate_spectrum(rx_v, fft_size=4096, avg=8)
        spec_h = estimate_spectrum(rx_h, fft_size=4096, avg=8)

        await asyncio.gather(
            consume_spectrum_console(spec_v, frames=12, topk=5),
            consume_spectrum_console(spec_h, frames=12, topk=5),
        )

    finally:
        # Ensure TX stops before device close (device close stops TX first, but this is explicit).
        if tx0 is not None:
            try:
                await tx0.stop_async()
            except Exception:
                pass

        try:
            dev.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
