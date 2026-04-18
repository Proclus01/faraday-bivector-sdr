from __future__ import annotations
from typing import Dict, Any, Optional

class DeviceController:
    def __init__(self, dev) -> None:
        self.dev = dev

    def set_rx_gain(self, channel: int, gain_db: float) -> bool:
        try:
            if hasattr(self.dev, "_sdr"):
                # Soapy path
                sdr = self.dev._sdr  # type: ignore
                sdr.setGain(0, channel, float(gain_db))  # 0=RX
            elif hasattr(self.dev, "set_rx_gain"):
                self.dev.set_rx_gain(channel, float(gain_db))  # type: ignore
            else:
                return False
            return True
        except Exception:
            return False

    def set_rx_center_freq(self, channel: int, cf_hz: float) -> bool:
        try:
            if hasattr(self.dev, "_sdr"):
                sdr = self.dev._sdr  # type: ignore
                sdr.setFrequency(0, channel, float(cf_hz))
            elif hasattr(self.dev, "set_rx_center_freq"):
                self.dev.set_rx_center_freq(channel, float(cf_hz))  # type: ignore
            else:
                return False
            return True
        except Exception:
            return False

    def set_rx_bandwidth(self, channel: int, bw_hz: float) -> bool:
        try:
            if hasattr(self.dev, "_sdr"):
                sdr = self.dev._sdr  # type: ignore
                sdr.setBandwidth(0, channel, float(bw_hz))
            elif hasattr(self.dev, "set_rx_bandwidth"):
                self.dev.set_rx_bandwidth(channel, float(bw_hz))  # type: ignore
            else:
                return False
            return True
        except Exception:
            return False

class ControlRegistry:
    def __init__(self) -> None:
        self._devs: Dict[str, DeviceController] = {}

    def register_device(self, devid: str, dev: Any) -> None:
        self._devs[devid] = DeviceController(dev)

    def get(self, devid: str) -> Optional[DeviceController]:
        return self._devs.get(devid)

GLOBAL_CONTROL = ControlRegistry()
