"""copyedit.py: spoken sentences rewritten as written English, and every
rewrite that changes a fact is refused."""
import re
import sys
import types

import pytest

import copyedit


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    # tests/conftest.py turns the copy edit off for every other test.
    monkeypatch.setattr(copyedit, "ENABLED", True)


def test_a_clean_grammar_fix_is_accepted():
    spoken = "It processes around 9,000 claims daily Agents uses a portal with requests going through Front Door"
    written = "It processes around 9,000 claims daily. Agents use a portal with requests going through Front Door."
    assert copyedit.problems(spoken, written) == []


def test_rewrites_that_change_a_fact_are_refused():
    # A tool dropped while the sentence is "tidied".
    assert copyedit.problems(
        "The jobs use Kafka for events and Redis for locks and Jenkins deploys them to AKS.",
        "The jobs use Kafka for events. They also use Redis for locks.")
    # A number lost.
    assert copyedit.problems("Backups are kept for 35 days in the vault.", "Backups are kept in the vault.")
    # An obligation softened, a "still" dropped.
    assert copyedit.problems("The failover runbook still needs an owner.",
                             "The failover runbook should have an owner.")
    # Who owns what, changed.
    assert copyedit.problems("Developers own the code and the platform team keeps it when it is infrastructure.",
                             "Developers own the code and the platform team maintains it.")
    # A chat reply instead of the sentence.
    assert copyedit.problems("Now let me show you the dashboard how it looks.", "Sure! Here's the dashboard:")
    # Words the speaker never said: the fact no longer matches its source.
    assert copyedit.problems("Then we have the logs tab where we can see the errors end to end like.",
                             "Then we have a logs tab where we can see the entire process, including end-to-end "
                             "errors.")
    # A new specific.
    assert copyedit.problems("The cache sits in front of the database.",
                             "The Redis cache sits in front of the PostgreSQL database.")


def test_a_safety_rule_is_never_rewritten():
    assert copyedit.problems("Never restart the primary database during business hours.",
                             "The primary database cannot be restarted during business hours.")


def test_only_sentences_that_read_as_speech_are_sent():
    assert copyedit.needs_edit("It processes 9,000 claims daily Agents uses a portal")         # no full stop
    assert copyedit.needs_edit("Then we have the alerts tab where the on-call engineer looks.")  # spoken opener
    assert copyedit.needs_edit("The jobs are Running on the cluster nightly.")                 # stray capital
    assert not copyedit.needs_edit("Check connection pool metrics before restarting services.")
    assert not copyedit.needs_edit("Prometheus and Grafana monitor health while PagerDuty sends alerts.")
    assert not copyedit.needs_edit("Never delete the state bucket so quickly.")                 # a safety rule
    assert not copyedit.needs_edit("Okay thanks.")


class _FakeProvider:
    def __init__(self, rewrite):
        self.rewrite = rewrite
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        items = re.findall(r"^(\d+)\. (.+)$", prompt.split("one item per line, and nothing else.")[-1], re.M)
        return "\n".join(f"{n}. {self.rewrite(text)}" for n, text in items)


def test_copyedit_keeps_accepted_rewrites_and_the_speakers_words_otherwise(monkeypatch):
    fixes = {
        "the service Runs on EKS right and it uses Aurora for orders":
            "The service runs on EKS and uses Aurora for orders.",
        "we keep the logs for 30 days in the bucket": "We keep the logs in the bucket.",   # loses 30
    }
    provider = _FakeProvider(lambda s: fixes.get(s, s))
    monkeypatch.setattr("llm_provider.get_llm_provider", lambda: provider)
    sentences = list(fixes) + ["Never delete the state bucket.", "Okay thanks."]
    edited, stats = copyedit.copyedit_sentences(sentences)
    assert edited == ["The service runs on EKS and uses Aurora for orders.",
                      "we keep the logs for 30 days in the bucket",
                      "Never delete the state bucket.", "Okay thanks."]
    assert stats["rewritten"] == 1 and stats["rejected"] == 1 and stats["model"] == "llm"
    # The safety rule and the two-word remark were not sent at all.
    assert "Never delete" not in provider.prompts[0] and "Okay thanks" not in provider.prompts[0]


def test_it_takes_at_most_a_quarter_of_a_kts_llm_call_limit(monkeypatch):
    provider = _FakeProvider(lambda s: s.capitalize() + ".")
    monkeypatch.setattr("llm_provider.get_llm_provider", lambda: provider)
    monkeypatch.setenv("LLM_MAX_CALLS_PER_KT", "8")
    monkeypatch.setattr(copyedit, "local_model_path", lambda: None)
    copyedit.copyedit_sentences([f"the service number {n} runs on the cluster" for n in range(100)])
    assert len(provider.prompts) == 2


def test_sentences_the_llm_could_not_take_go_to_the_local_model(monkeypatch):
    class RateLimited:
        calls = 0

        def generate(self, prompt, **kwargs):
            RateLimited.calls += 1
            if RateLimited.calls > 1:
                raise RuntimeError("429 rate limit")
            items = re.findall(r"^(\d+)\. (.+)$", prompt.split("nothing else.")[-1], re.M)
            return "\n".join(f"{n}. {t[0].upper()}{t[1:]}." for n, t in items)

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

        def create_chat_completion(self, messages, **kwargs):
            spoken = messages[-1]["content"]
            return {"choices": [{"message": {"content": spoken[0].upper() + spoken[1:] + "."}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("llm_provider.get_llm_provider", lambda: RateLimited())
    monkeypatch.setattr(copyedit, "local_model_path", lambda: "model.gguf")
    monkeypatch.setattr(copyedit, "available_memory_gb", lambda: 8.0)
    sentences = [f"the worker number {n} runs on the cluster" for n in range(20)]   # two batches of 15 + 5
    edited, stats = copyedit.copyedit_sentences(sentences)
    assert edited == [s[0].upper() + s[1:] + "." for s in sentences]   # the first 15 by the LLM, the rest locally
    assert stats["model"] == "llm + model.gguf" and stats["rewritten"] == 20


def test_the_local_model_is_freed_after_use_and_stops_at_its_time_limit(monkeypatch):
    made, freed, clock = [], [], [1000.0]

    class FakeLlama:
        def __init__(self, model_path, n_ctx, n_threads, verbose):
            made.append(n_threads)

        def create_chat_completion(self, messages, **kwargs):
            clock[0] += 40.0                 # each sentence takes 40 s
            spoken = messages[-1]["content"]
            return {"choices": [{"message": {"content": spoken[0].upper() + spoken[1:] + "."}}]}

        def __del__(self):
            freed.append(True)

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("llm_provider.get_llm_provider", lambda: None)
    monkeypatch.setattr(copyedit, "local_model_path", lambda: "model.gguf")
    monkeypatch.setattr(copyedit, "MAX_SECONDS", 100.0)
    monkeypatch.setattr(copyedit, "available_memory_gb", lambda: 8.0)
    monkeypatch.setattr(copyedit.time, "time", lambda: clock[0])
    sentences = [f"the worker number {n} runs on the cluster" for n in range(5)]
    edited, stats = copyedit.copyedit_sentences(sentences)
    assert edited[:3] == [s[0].upper() + s[1:] + "." for s in sentences[:3]]   # 0 s, 40 s, 80 s: started in time
    assert edited[3:] == sentences[3:]                                       # after 100 s: left as said
    assert made == [copyedit.THREADS] and freed == [True]
    assert stats["model"] == "model.gguf" and stats["rewritten"] == 3


def test_the_local_model_takes_over_when_every_llm_call_fails(monkeypatch):
    class DeadKey:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("401 invalid API key")

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

        def create_chat_completion(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "The service runs on EKS."}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("llm_provider.get_llm_provider", lambda: DeadKey())
    monkeypatch.setattr(copyedit, "local_model_path", lambda: "model.gguf")
    monkeypatch.setattr(copyedit, "available_memory_gb", lambda: 8.0)
    edited, stats = copyedit.copyedit_sentences(["so the service Runs on EKS right"])
    assert edited == ["The service runs on EKS."] and stats["model"] == "model.gguf"


def test_the_local_model_is_not_loaded_when_memory_is_short(monkeypatch):
    loaded = []

    class FakeLlama:
        def __init__(self, *a, **k):
            loaded.append(True)

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("llm_provider.get_llm_provider", lambda: None)
    monkeypatch.setattr(copyedit, "local_model_path", lambda: "model.gguf")
    monkeypatch.setattr(copyedit, "available_memory_gb", lambda: 0.9)
    text = ["so yeah the service Runs on EKS right"]
    edited, stats = copyedit.copyedit_sentences(text)
    assert edited == text and loaded == [] and stats["skipped"] == "low memory"
    assert copyedit.available_memory_gb.__name__ == "<lambda>"   # the real reading is patched out here
    monkeypatch.undo()
    assert copyedit.available_memory_gb() is None or copyedit.available_memory_gb() > 0


def test_without_a_model_nothing_changes(monkeypatch):
    monkeypatch.setattr("llm_provider.get_llm_provider", lambda: None)
    monkeypatch.setattr(copyedit, "local_model_path", lambda: None)
    text = ["so yeah the service Runs on EKS right"]
    edited, stats = copyedit.copyedit_sentences(text)
    assert edited == text and stats["model"] is None and stats["rewritten"] == 0


def test_rewrites_that_change_who_said_it_its_tense_or_its_topic_are_refused():
    """Seen on the local model (audit 9aam): each kept every content word."""
    refused = {
        # A rule became history.
        "No changes in the last week of December.":
            "There have been no changes in the last week of December.",
        # Owen's answer became Nadia's.
        "Nadia: And my first week? Owen: Get access to the Azure subscription and the on-prem jump host.":
            "Nadia: My first week involves getting access to the Azure subscription and the on-prem jump host.",
        # "deployed" was not said, and moved the fact to Deployment.
        "Java Spring Boot services on Red Hat OpenShift in our two data centres.":
            "Java Spring Boot services are deployed on Red Hat OpenShift in two data centers.",
    }
    for spoken, rewrite in refused.items():
        assert copyedit.problems(spoken, rewrite), rewrite
    accepted = {
        "For a rollback you just revert the commit in the config repo and Argo puts the old version back.":
            "For a rollback, you revert the commit in the config repo, and Argo puts the old version back.",
        "Nadia: How will I know something is wrong? Owen: Azure Monitor alerts on failed bookings.":
            "Nadia: How can I know if something is wrong? Owen: Azure Monitor alerts on failed bookings.",
    }
    for spoken, rewrite in accepted.items():
        assert copyedit.problems(spoken, rewrite) == [], rewrite


def test_a_correction_marker_may_go_but_a_word_that_only_looks_like_one_may_not():
    assert copyedit.problems("We use Jenkins, sorry, GitHub Actions for the builds.",
                             "We use Jenkins, or rather GitHub Actions, for the builds.") == []
    # Still refused: the corrected tool is a tool, and it is gone.
    assert copyedit.problems("We use Jenkins, sorry, GitHub Actions for the builds.",
                             "We use GitHub Actions for the builds.")
    # "mean" and "wait" are content outside a marker.
    assert copyedit.problems("The mean latency is 200 ms.", "The latency is 200 ms.")
    assert copyedit.problems("Wait five minutes before retrying the job.", "Retry the job after five minutes.")


def test_a_rewrite_may_not_drop_the_word_that_withdraws_a_value():
    # A user's KT (23 Oct): "The database backups are retained for seven
    # days—actually, the new retention policy is 30 days." was copy edited to
    # "... seven days. The new retention policy is 30 days.", and the LLM then
    # listed "Database backups retained for seven days" as current.
    from copyedit import problems

    said = "The database backups are retained for seven days—actually, the new retention policy is 30 days."
    assert "correction lost" in problems(said, "The database backups are retained for seven days. "
                                               "The new retention policy is 30 days.")
    assert "correction lost" not in problems(said, "The database backups are retained for seven days. "
                                                   "Actually, the new retention policy is 30 days.")
