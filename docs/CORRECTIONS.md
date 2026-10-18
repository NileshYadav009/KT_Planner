# Corrections, changes and contradictions

How the pipeline decides which value a KT document shows when the speaker
gave more than one. All of it is deterministic: it runs with or without an
LLM, before the copy edit, and the Sources appendix always quotes what was
said. Evidence and history: `docs/REPOSITORY_AUDIT.md` §9aan, §9aap, §9aaq.
Cases: `evaluation/matrix/cases.json` (categories A and D), run by
`python evaluation/matrix.py LABEL`. Unit tests:
`tests/test_devops_transcription.py` (same sentence, next sentence, further
back), `tests/test_conflicts.py` (conflicts), `tests/test_knowledge_builders.py`
(plans and connection quotes).

## Principles

1. The document shows the speaker's latest valid statement. A withdrawn value
   is never shown on its own as current.
2. A value is replaced only when the session says so: a correction marker, a
   statement that the old value is out of date, or a pointer back to it.
   Saying a different value later is not enough: that is a possible conflict,
   and it is flagged, not resolved.
3. Different environments, services or times may have different values.
   Those are distinct facts, never corrections of each other.
4. What was withdrawn stays in the evidence: the transcript and the Sources
   appendix quote it as said. A correction folded into the sentence before
   it is recorded in the fact ledger as excluded, with its reason; nothing
   else is removed from the document.
5. Nothing in a transcript is executed or verified. A command is shown as
   said. A value is never described as checked against the system.

## Rules

| What was said | What the document shows | Code |
|---|---|---|
| A number corrected in the same sentence: "the limit is 4000, sorry, 2000"; "2000K—no, sorry, it's 3000K"; "every 30 minutes—sorry, every 30 seconds" | The corrected value only: "the limit is 2000" | `devops_transcription.resolve_self_corrections` |
| The next sentence only restates the number: "The limit is 500 requests a minute. Sorry, not 500, the limit is 300 requests a minute." | The first sentence with the corrected number. Both sentences are cited, and the ledger records the correction as excluded, with its reason. | `resolve_correction_across`, `context_mapper.merge_spoken_corrections` |
| The next sentence replaces the value and says more: "We use PostgreSQL 14. Actually, the upgrade is complete; production is now on PostgreSQL 16." | Both sentences together, as said, as one unit in one section. The old value is never on its own. | `replaces_value_before`, `context_mapper.segment_sentences` |
| A later sentence points back: "I was wrong about the backups earlier, they are kept for 35 days now." | The earlier sentence is moved next to the correction and shown with it. Only when the earlier sentence states nothing else, is about the named topic, and is at most 20 sentences back. | `points_back_to`, `segment_sentences` |
| A tool replaced: "we moved off PagerDuty to Opsgenie" | Tool fields and the Technology summary name the new tool | `conflicts.find_superseded_tools`, `knowledge_builder.apply_conflicts` |
| A single-valued field stated twice (RTO, RPO, an owner) | The corrected value, with what it replaces; with no correction marker, "Conflicting values stated" to confirm | `conflicts.resolve_single_value`, `field_populator.extract_rto_rpo` |
| Two counts of one thing, with no correction: "two availability zones" ... "three availability zones"; or one setting with two values: "The upload size limit is 2000K." ... "The upload size limit is 3000K." | Both statements, and a "Possible conflict" warning in the section and the knowledge gaps. Not for different environments ("Staging runs 2 replicas", "Production runs 6"), different named parts ("the orders API timeout", "the payments API timeout"), a value with a condition ("800 ms during peak"), a volume ("500 orders a minute"), a past value or a plan. | `conflicts.find_value_conflicts` |
| A rule permitted and forbidden: "Deploying on Fridays is fine" ... "Never deploy on Fridays" | Both, with a "Possible conflict" warning. Only when the permitting sentence names the forbidden action (its verb and most of its words): "we can always redeploy an older model" does not conflict with "don't delete the old model versions". | `conflicts.find_contradictions` |
| A contradiction of meaning, with an LLM configured: "Developers own deployment approvals" ... "Only the release manager can approve production deployments"; "The service is public" ... "Only private subnets can reach it" | Both, with a "Possible conflict" warning. One call per KT lists the statements that have a close neighbour by meaning in another unit; the model names contradicting pairs and may only name listed statements. | `conflicts.find_conflicts_with_llm`, `knowledge_builder.apply_llm_conflict_check` |
| A plan or a guess naming a technology: "planning to migrate to Kubernetes", "might be using Redis Cluster" | The sentence as said; the technology is not listed as a current component or in the Technology summary | `field_populator.current_tool_mentions` |

## What counts as a correction signal

- **Same sentence:** "sorry", "I mean", "or rather", "no wait", "correction",
  "scratch that", "my mistake", optionally after "no,", a comma, a dash or a
  pause. A bare "no" or "actually" is not enough ("9 to 5, no 24/7 support";
  "5 minutes, actually 10 on a bad day").
- **Next sentence:** a sentence opening with one of the markers above, "wait",
  "forget what I said" or "I was wrong"; or one that says the old value "is
  old", "is outdated", "is wrong", "use the latter", "the correct value is".
  An opening "actually" counts only with a word of change ("now",
  "upgraded", "moved", "changed"): "Actually, the rollback takes 10
  minutes" adds a fact.
- **Further back:** "I was wrong about X", "forget what I said about X",
  "what I said about X". X must name the earlier sentence's topic.
- **The values must be of one kind:** the same unit after the number ("7
  days" / "30 days", "2000K" / "3000K") or the same word naming it ("port
  8080" / "port is 8443", "PostgreSQL 14" / "PostgreSQL 16").

## Not handled yet

- Without an LLM, a contradiction of meaning ("Developers own deployment
  approvals" / "Only the release manager can approve production
  deployments") is shown but not flagged; so is "Backups run hourly" / "the
  backup interval is every six hours". With an LLM, the conflict check flags
  them.
- The LLM check sees only statements with a close neighbour by meaning (on
  the dev KTs, 18 to 31 statements, from KTs of 27 to 55 units). A contradiction between two
  statements that read as unrelated can be missed.
- A path or identifier corrected with a bare "no" ("/api/v1/orders, no,
  /api/v2/orders"): shown as said, both values in one sentence.
