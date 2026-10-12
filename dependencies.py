"""Dependency explorer (P3-4): what the systems of a workspace have in common.

Each finished KT's architecture picture (P1-9) places the system's
components: its data stores and queues, the outside services it calls, how
it is delivered and how it is operated. Read across every KT of a
workspace, the same component under several systems is a dependency to
know about when it changes or fails: "Kafka is named by Relay, Beacon and
Kestrel". A component name is not an instance, so two systems naming Redis
may use two different Redis clusters; the explorer says what the KTs name
and quotes the sentence, it does not claim more.

Service catalogs (Backstage, Cortex) hold curated ownership; this is the
view from the KTs themselves, available without one.
"""
import re
from typing import Any, Dict, List, Optional

# Architecture-graph groups, as dependencies (what the system needs to run)
# or tooling (how it is built and operated).
DEPENDENCY_GROUPS = {"data": "data store or queue", "external": "outside service", "entry": "entry point"}
TOOLING_GROUPS = {"delivery": "delivery", "operations": "operations"}
_GENERIC = {"users", "customers", "clients", "browser", "mobile app"}


def _key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _graph(knowledge_object: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for section in knowledge_object.get("sections") or []:
        if section.get("id") == "architecture_reference" and isinstance(section.get("_architecture_graph"), dict):
            return section["_architecture_graph"]
    return None


def system_entry(job_id: str, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One KT's components by group, with the stated connection for each."""
    ko = job.get("knowledge_object") or {}
    graph = _graph(ko)
    if not graph:
        return None
    stated = {}
    for edge in graph.get("edges") or []:
        other = edge.get("to") if edge.get("from") == "central" else edge.get("from")
        stated.setdefault(other, {"label": edge.get("label"), "quote": edge.get("quote")})
    components = []
    for group, kind in {**DEPENDENCY_GROUPS, **TOOLING_GROUPS}.items():
        for node in graph.get(group) or []:
            label = str(node.get("label") or "").strip()
            if not label or _key(label) in _GENERIC:
                continue
            components.append({"name": label, "group": group, "kind": node.get("caption") or kind,
                               "dependency": group in DEPENDENCY_GROUPS, "stated": stated.get(node.get("id"))})
    central = graph.get("central") or {}
    return {"job_id": job_id, "system": ko.get("system_name") or job.get("title") or "Untitled KT",
            "platform": central.get("header"), "cloud": (graph.get("boundary") or {}).get("label"),
            "components": components}


def explore(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Components named by more than one system, dependencies first.
    `entries` newest first: a system with several KTs counts once, by its
    newest KT."""
    newest: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        newest.setdefault(_key(entry["system"]), entry)
    entries = list(newest.values())
    index: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        for comp in entry["components"]:
            item = index.setdefault(_key(comp["name"]), {"component": comp["name"], "kind": comp["kind"],
                                                         "dependency": comp["dependency"], "systems": []})
            if any(s["job_id"] == entry["job_id"] for s in item["systems"]):
                continue
            item["dependency"] = item["dependency"] or comp["dependency"]
            item["systems"].append({"job_id": entry["job_id"], "system": entry["system"],
                                    "how": (comp["stated"] or {}).get("label"),
                                    "quote": (comp["stated"] or {}).get("quote")})
    shared = [i for i in index.values() if len(i["systems"]) > 1]
    shared.sort(key=lambda i: (not i["dependency"], -len(i["systems"]), i["component"].lower()))
    return {"systems": [{k: e[k] for k in ("job_id", "system", "platform", "cloud")}
                        | {"components": len(e["components"])} for e in entries],
            "shared": shared}
