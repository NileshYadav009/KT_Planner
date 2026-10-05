"""The tenant a KT job runs for, and its LLM policy (P0-9).

Set by pipeline.run_kt_pipeline for the duration of a job; worker threads
inherit it through contextvars.copy_context(), the same way usage tracking
does. Read by the LLM cache (entries are tenant-scoped) and by the provider
wrapper (a tenant whose policy is "none" sends nothing to an external LLM).
"""
import contextvars
from contextlib import contextmanager

CURRENT_TENANT: contextvars.ContextVar[str] = contextvars.ContextVar("continuum_tenant", default="")
LLM_POLICY: contextvars.ContextVar[str] = contextvars.ContextVar("continuum_llm_policy", default="default")


class LLMDisabledForTenant(RuntimeError):
    """The tenant does not allow transcript content to reach an external LLM."""


def llm_allowed() -> bool:
    return LLM_POLICY.get() != "none"


@contextmanager
def tenant_scope(tenant_id: str, llm_policy: str = "default"):
    t1 = CURRENT_TENANT.set(tenant_id or "")
    t2 = LLM_POLICY.set(llm_policy or "default")
    try:
        yield
    finally:
        LLM_POLICY.reset(t2)
        CURRENT_TENANT.reset(t1)
