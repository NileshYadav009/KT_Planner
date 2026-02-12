from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel
import os
import json
import uuid
from datetime import datetime

router = APIRouter()

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
INDEX_PATH = os.path.join(TEMPLATES_DIR, "index.json")
AUDIT_LOG = os.path.join(TEMPLATES_DIR, "audit_logs.jsonl")

os.makedirs(TEMPLATES_DIR, exist_ok=True)

def _read_index():
    if not os.path.exists(INDEX_PATH):
        return {}
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_index(idx):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)

def _append_audit(entry: dict):
    entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

class TemplateIn(BaseModel):
    title: str
    business_unit: str | None = None
    schema: dict
    description: str | None = None


@router.get("/")
def list_templates():
    idx = _read_index()
    out = []
    for tid, meta in idx.items():
        out.append({"id": tid, "title": meta.get("title"), "versions": meta.get("versions", []), "locked": meta.get("locked", False), "business_unit": meta.get("business_unit")})
    return {"templates": out}


@router.post("/")
def create_template(payload: TemplateIn, x_user: str | None = Header(None), x_user_role: str | None = Header(None)):
    # simple RBAC: only Admin/Manager can create templates
    if x_user_role not in {"Admin", "Manager"}:
        raise HTTPException(status_code=403, detail="insufficient role to create templates")

    tid = str(uuid.uuid4())
    version = 1
    filename = f"{tid}_v{version}.json"
    filepath = os.path.join(TEMPLATES_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"id": tid, "version": version, "title": payload.title, "business_unit": payload.business_unit, "schema": payload.schema, "description": payload.description, "created_by": x_user}, f, indent=2)

    idx = _read_index()
    idx[tid] = {"title": payload.title, "business_unit": payload.business_unit, "versions": [filename], "locked": False}
    _write_index(idx)

    _append_audit({"action": "create_template", "template_id": tid, "version": version, "user": x_user, "role": x_user_role})

    return {"id": tid, "version": version}


@router.get("/{template_id}")
def get_template(template_id: str, version: int | None = None):
    idx = _read_index()
    if template_id not in idx:
        raise HTTPException(status_code=404, detail="template not found")
    versions = idx[template_id].get("versions", [])
    if not versions:
        raise HTTPException(status_code=404, detail="no versions available")
    if version is None:
        filename = versions[-1]
    else:
        filename = next((v for v in versions if v.endswith(f"_v{version}.json")), None)
        if filename is None:
            raise HTTPException(status_code=404, detail="version not found")

    filepath = os.path.join(TEMPLATES_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.post("/{template_id}/lock")
def lock_template(template_id: str, action: dict, x_user: str | None = Header(None), x_user_role: str | None = Header(None)):
    # action: {"op": "lock"} or {"op": "unlock"}
    if x_user_role not in {"Admin", "Manager"}:
        raise HTTPException(status_code=403, detail="insufficient role to lock templates")
    idx = _read_index()
    if template_id not in idx:
        raise HTTPException(status_code=404, detail="template not found")

    op = action.get("op")
    if op == "lock":
        idx[template_id]["locked"] = True
        _append_audit({"action": "lock_template", "template_id": template_id, "user": x_user, "role": x_user_role})
    elif op == "unlock":
        idx[template_id]["locked"] = False
        _append_audit({"action": "unlock_template", "template_id": template_id, "user": x_user, "role": x_user_role})
    else:
        raise HTTPException(status_code=400, detail="invalid op")

    _write_index(idx)
    return {"id": template_id, "locked": idx[template_id]["locked"]}


@router.get("/audit/logs")
def get_audit_logs(limit: int = 200):
    logs = []
    if os.path.exists(AUDIT_LOG):
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        continue
    return {"count": len(logs), "logs": logs[-limit:]}
