import json
import os
from typing import Any, Dict

DEFAULT_POLICY_FILE = "policy.json"

DEFAULT_POLICY = {
    "confidence_accept_threshold": 0.8,
    "implementation_indicators": [
        "kubectl","docker","compose","helm","apply -f","systemctl","service",
        "install ","pip install","npm install","sh ","bash ","curl ",".yaml",".yml",
        "deployment.yaml","step ","1.","2.","run ","execute ","sudo "
    ],
    "conceptual_sections": ["overview","architecture","data flow","monitoring","risks","faqs"]
}


def load_policy(path: str = DEFAULT_POLICY_FILE) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            pass
    return DEFAULT_POLICY.copy()


def save_policy(policy: Dict[str, Any], path: str = DEFAULT_POLICY_FILE) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(policy, f, indent=2)
        return True
    except Exception:
        return False
