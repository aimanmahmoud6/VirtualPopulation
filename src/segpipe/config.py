# =============================================================================
# Configuration Loader
#
# Summary:
#   Loads YAML configuration files and expands vertebra ranges into organ list.
#
# Created by Aiman Mahmoud, King's College London, February 2026.
# Email: aiman.2.mahmoud@kcl.ac.uk
# =============================================================================

from dataclasses import dataclass
from typing import Any, Dict, List
import yaml

@dataclass
class Config:
    params: Dict[str, Any]
    organs: List[Dict[str, Any]]
    qc: Dict[str, Any]

def _expand_vertebrae(org_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    organs = list(org_cfg.get("organs", []))
    ranges = org_cfg.get("vertebrae_ranges", [])
    tpl = org_cfg.get("vertebrae_template", {"name_prefix": "Vertebra_", "method": "gaussian", "param": 1.0})

    for r in ranges:
        a, b = int(r[0]), int(r[1])
        for v in range(a, b + 1):
            organs.append({
                "id": int(v),
                "name": f"{tpl.get('name_prefix','Vertebra_')}{v}",
                "method": tpl.get("method", "gaussian"),
                "param": tpl.get("param", 1.0),
            })
    return organs

def load_config(params_path: str, organs_path: str, qc_path: str) -> Config:
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    with open(organs_path, "r", encoding="utf-8") as f:
        org_cfg = yaml.safe_load(f)
    organs = _expand_vertebrae(org_cfg)

    with open(qc_path, "r", encoding="utf-8") as f:
        qc = yaml.safe_load(f)

    return Config(params=params, organs=organs, qc=qc)
