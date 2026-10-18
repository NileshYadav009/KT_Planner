"""Adversarial probe KT (written 18 Oct before any output was seen), no LLM.

    python evaluation/adversarial.py

Each probe: a quote and the sections a careful KT writer would accept.
Prints where each quote landed and the coverage table. Used to find
systematic defects, not to tune: a fix must be generic and be measured on the
dev/holdout/blind sets.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
state = tempfile.mkdtemp(prefix="adv-")
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(state, "c.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(state, "g.json"),
    "KT_ASSETS_DIR": os.path.join(state, "a"),
    "CONTINUUM_AUTH": "disabled", "CONTINUUM_COPYEDIT": "0",
    "LLM_CACHE_PATH": os.path.join(state, "llm.sqlite"),
})
for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ[key] = ""
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")
import enccache  # noqa: E402
enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

TRANSCRIPT = """Kiran: This is the handover for Harbor, the shipment tracking service our logistics customers use to follow their parcels.
Harbor is a set of Go services on Amazon ECS behind an Application Load Balancer, and the tracking events land in Amazon Aurora PostgreSQL.
Carrier updates come in through an SQS queue, and a worker service writes them to Aurora.
We deploy with GitHub Actions: a merge to main builds the image and pushes it to ECR, and then the pipeline updates the ECS service.
To roll back, we redeploy the previous task definition from the ECS console.
Grafana shows queue depth and API latency, and an alert fires when the SQS queue is older than ten minutes.
When that alert fires, scale the worker service to six tasks and check the carrier API status page.
The platform team owns the ECS cluster, and my team owns the Go services.
If something breaks at night, page the on-call through PagerDuty, and if they do not answer in fifteen minutes, call Maria, the engineering manager.
Right now we run in one region, us-east-1.
Next quarter we plan to add a second region in eu-west-1, but nobody has started on it.
Aurora snapshots are taken every night and kept for seven days.
We have never tested a restore.
Never run the database migration script against production without a snapshot first, it drops the old columns.
The Black Friday week is the busiest time, so there are no deployments in that week.
Customer data is encrypted at rest with KMS, and only the support team can see parcel recipient names.
The biggest cost is Aurora, about three thousand dollars a month.
The API rate limit is 500 requests a minute per customer.
Sorry, not 500, the limit is 300 requests a minute per customer.
On your first day, get access to the AWS account and the PagerDuty rota.
In your first month, you should take over the weekly release.
"""

PROBES = [
    ("Go services on Amazon ECS behind an Application Load Balancer", ["architecture_reference", "system_overview"]),
    ("a worker service writes them to Aurora", ["architecture_reference", "system_overview"]),
    ("a merge to main builds the image and pushes it to ECR", ["deployment_and_rollback"]),
    ("redeploy the previous task definition", ["deployment_and_rollback"]),
    ("Grafana shows queue depth and API latency", ["monitoring_observability"]),
    ("an alert fires when the SQS queue is older than ten minutes", ["monitoring_observability"]),
    ("scale the worker service to six tasks", ["common_failures", "monitoring_observability"]),
    ("The platform team owns the ECS cluster", ["ownership_escalation"]),
    ("page the on-call through PagerDuty", ["ownership_escalation"]),
    ("call Maria, the engineering manager", ["ownership_escalation"]),
    ("we run in one region, us-east-1", ["architecture_reference", "system_overview", "environments", "disaster_recovery"]),
    ("we plan to add a second region in eu-west-1", ["open_responsibilities", "disaster_recovery"]),
    ("Aurora snapshots are taken every night", ["disaster_recovery"]),
    ("We have never tested a restore", ["disaster_recovery"]),
    ("Never run the database migration script against production", ["danger_zones"]),
    ("The Black Friday week is the busiest time", ["known_bad_days"]),
    ("Customer data is encrypted at rest with KMS", ["security_controls"]),
    ("only the support team can see parcel recipient names", ["security_controls"]),
    ("The biggest cost is Aurora", ["cost_optimization"]),
    ("get access to the AWS account and the PagerDuty rota", ["day1_survival_checklist"]),
    ("take over the weekly release", ["first_30_day_ownership"]),
]

import ai  # noqa: E402
import golden_eval  # noqa: E402
import llm_provider  # noqa: E402
import pipeline  # noqa: E402
from context_mapper import ContextMappingPipeline  # noqa: E402
from devops_transcription import clean_transcript  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402

pipeline.get_llm_provider = ai.get_llm_provider = llm_provider.get_llm_provider = lambda: None
pipeline.record_candidates = lambda *a, **k: None
pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
result = pipeline.run_kt_pipeline("adv", clean_transcript(TRANSCRIPT))
ko = result["knowledge_object"]
texts = golden_eval.section_texts(ko)
norm = golden_eval.norm
right = 0
for quote, allowed in PROBES:
    q = norm(quote)
    found = sorted(sid for sid, items in texts.items() if any(q in norm(t) for t in items))
    ok = bool(set(found) & set(allowed))
    right += ok
    print(("ok   " if ok else "WRONG"), quote[:60].ljust(60), "->", found or "LOST", "" if ok else f"(want {allowed})")
print(f"{right}/{len(PROBES)}")
print("--- ownership section:")
for t in texts.get("ownership_escalation", []):
    print("   ", t[:200])
print("--- limit lines:")
for sid, items in texts.items():
    for t in items:
        if "requests a minute" in t:
            print("  ", sid, "|", t[:200])
print("--- sources quoting the limit:")
for src in ko.get("sources") or []:
    if "requests a minute" in (src.get("quote") or ""):
        print("   ", src.get("quote"))
print("--- coverage:")
for sec in ko.get("sections") or []:
    if sec.get("id") == "kt_coverage":
        for r in sec.get("_coverage_rows") or []:
            print("  ", r.get("Domain"), "|", r.get("Coverage"), "|", (r.get("Assessment") or "")[:150])
summary = next((s.get("_knowledge_coverage_summary") for s in ko.get("sections") or [] if s.get("id") == "kt_coverage"), None)
print("ledger summary:", json.dumps(summary))
pipeline.JOB_STORE.delete("adv")
