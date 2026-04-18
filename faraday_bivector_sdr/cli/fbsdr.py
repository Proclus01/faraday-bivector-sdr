from __future__ import annotations
import sys
import asyncio
from ..application.orchestrator import run_config
from ..runtime.webui import _enumerate_devices

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m faraday_bivector_sdr.cli.fbsdr <config.yaml> | devices")
        sys.exit(1)
    if sys.argv[1] == "devices":
        info = _enumerate_devices()
        print(info)
        return
    cfg_path = sys.argv[1]
    asyncio.run(run_config(cfg_path))

if __name__ == "__main__":
    main()
