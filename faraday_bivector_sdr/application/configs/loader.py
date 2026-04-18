from __future__ import annotations
import yaml
from .schema import Config, DeviceDef, ProjectionDef, PipelineDef, OutputDef

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    devices = [DeviceDef(**d) for d in data.get("devices", [])]
    projections = [ProjectionDef(**p) for p in data.get("projections", [])]
    pipelines = [PipelineDef(**p) for p in data.get("pipelines", [])]
    outputs = [OutputDef(type=o.get("type"), params={k: v for k, v in o.items() if k != "type"}) for o in data.get("outputs", [])]
    return Config(devices=devices, projections=projections, pipelines=pipelines, outputs=outputs)
