# Corrections, changes and contradictions

How the pipeline decides which value a KT document shows when the speaker
gave more than one. All of it is deterministic: it runs with or without an
LLM, before the copy edit, and the Sources appendix always quotes what was
said. Evidence and history: `docs/REPOSITORY_AUDIT.md` §9aan, §9aap, §9aaq, §9aar, §9aas, §9aav.
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
| The next sentence states a change, with no marker: "The database is PostgreSQL 14. We upgraded production to PostgreSQL 16 last night."; "It is now 40 connections." | Both together, as said (as above). | `replaces_value_before` (a change verb with "to", "is now") |
| The next sentence corrects a statement with no number: "The service is owned by Platform Engineering. Actually, application ownership belongs to Payments, but Platform owns the cluster." | Both together, as said. It opens with a correction marker or "actually" (not "actually, that reminds me") and shares a content word with the sentence before. | `corrects_statement_before` |
| A change said with a hedge: "The backup retention is seven days. I think we changed it to thirty days, but I'm not sure." | Both together, an "Unconfirmed change — confirm with the outgoing owner" warning, and a knowledge-gap line. The section still counts as discussed. | `conflicts.find_unconfirmed_changes` |
| A value withdrawn in the same sentence that says more than the new value: "The database backups are retained for seven days—actually, the new retention policy is 30 days."; "PostgreSQL 14—sorry, we upgraded it last week, so it's PostgreSQL 16 now." | The sentence whole, as said: never cut at the marker into two facts. An LLM answer that shows the old value without the new one is refused (an extracted value or line is dropped; a polish or copy edit falls back to the sentence as said). | `devops_transcription.withdrawn_values`, `context_mapper.split_facts`, `grounding.ground_structured`, `ai._polish_one`, `copyedit.problems` |
| One value compared across environments in one sentence: "Staging uses 2000K, while production uses 3000K." | One statement, not cut into two facts; no conflict. | `compares_one_value` in `context_mapper.split_facts` |
| A later sentence points back: "I was wrong about the backups earlier, they are kept for 35 days now." | The earlier sentence is moved next to the correction and shown with it. Only when the earlier sentence states nothing else, is about the named topic, and is at most 20 sentences back. | `points_back_to`, `segment_sentences` |
| A later correction says again what it withdraws, in one sentence or two: "Actually, I said earlier we deploy on Tuesdays. That's wrong, it's Wednesdays now."; "Earlier I said the cluster has three nodes; that's not right, it has five." | The earlier sentence is moved next to the correction and shown with it, under the same conditions: it states nothing else, it is at most 20 sentences back, and at least two of the restated words (two thirds of them) are its own. The restatement and its withdrawal stay together either way. | `withdraws_said_before`, `restates`, `segment_sentences` |
| A day of the week corrected: "Releases go out on Thursdays. Sorry, Fridays."; "every Sunday, sorry, every Saturday" | As a number: the corrected day only, or both together when the correction says more. | `_CORRECTED_VALUE` (`_WEEKDAY`) |
| A tool replaced: "we moved off PagerDuty to Opsgenie" | Tool fields and the Technology summary name the new tool | `conflicts.find_superseded_tools`, `knowledge_builder.apply_conflicts` |
| A single-valued field stated twice (RTO, RPO, an owner) | The corrected value, with what it replaces; with no correction marker, "Conflicting values stated" to confirm | `conflicts.resolve_single_value`, `field_populator.extract_rto_rpo` |
| Two counts of one thing, with no correction: "two availability zones" ... "three availability zones"; or one setting with two values: "The upload size limit is 2000K." ... "The upload size limit is 3000K." | Both statements, and a "Possible conflict" warning in the section and the knowledge gaps. Not for different environments ("Staging runs 2 replicas", "Production runs 6"), different named parts ("the orders API timeout", "the payments API timeout"), a value with a condition ("800 ms during peak"), a volume ("500 orders a minute"), a past value or a plan. | `conflicts.find_value_conflicts` |
| A rule permitted and forbidden: "Deploying on Fridays is fine" ... "Never deploy on Fridays" | Both, with a "Possible conflict" warning. Only when the permitting sentence names the forbidden action (its verb and most of its words): "we can always redeploy an older model" does not conflict with "don't delete the old model versions". | `conflicts.find_contradictions` |
| A contradiction of meaning, with an LLM configured: "Developers own deployment approvals" ... "Only the release manager can approve production deployments"; "The service is public" ... "Only private subnets can reach it" | Both, with a "Possible conflict" warning. One call per KT lists the statements that have a close neighbour by meaning in another unit; the model names contradicting pairs and may only name listed statements. | `conflicts.find_conflicts_with_llm`, `knowledge_builder.apply_llm_conflict_check` |
| A value given on an old document's word, or doubted in the next sentence: "Our recovery point objective is four hours, or at least that is what the previous document says. I need to confirm whether this target is still valid."; "... memory limit is 4 GB or 8 GB. That value needs confirmation ..." | Both sentences together, as said, on the follow-up list to confirm. The value fills no field (no "RPO: four hours"). A sentence with a claim or a gap of its own is not joined. | `dialogue._DOUBTED_SOURCE_RE`, `doubts_statement_before`, `is_field_candidate` |
| A count said to be out of date: "Earlier, someone mentioned four production replicas, but that information is outdated." | As said; no conflict with the current count. Nor is "three replicas in staging" a conflict with "production runs five replicas" (the environment can follow the count). | `conflicts._PAST_RE`, `conflicts._counts` |
| A plan or a guess naming a technology: "planning to migrate to Kubernetes", "might be using Redis Cluster" | The sentence as said; the technology is not listed as a current component or in the Technology summary | `field_populator.current_tool_mentions` |

## Between speakers (a conversation)

With two or more speakers (pasted "Name:" turns, or a diarised recording),
facts come from the giver and anyone who is not a receiver; the receiver's
words count only where the giver confirmed them (`speaker_roles.py`). The
giver is the speaker who states the most, unless the user chose another on
the KT page. A receiver is a speaker whose sentences are mostly questions.

| What was said | What the document shows | Code |
|---|---|---|
| The receiver guesses, the giver turns it down: "So it stores invoices in MySQL, right?" / "No, it's PostgreSQL on Amazon RDS." | The question and the answer together, as said. "No" with the value is not a gap; the guessed tool is not a current component. In a conversation a tag question stays a question. | `speaker_roles.apply_roles`, `dialogue._NEGATIVE_ANSWER_RE`, `field_populator.current_tool_mentions`, `devops_transcription.remove_fillers` |
| The receiver states something, the giver corrects it: "I heard backups are kept for 30 days." / "Actually, backups are kept for 14 days, not 30." | The giver's sentence only. The receiver's is left out, recorded "said by Meera; corrected by Arjun". | `apply_roles` |
| The receiver corrects the giver, and the giver agrees: "The cluster runs three nodes." / "I think it's five now." / "Oh right, yes, it's five nodes since the scaling." | The agreement only (with the receiver's words when the agreement does not repeat the point: "Yes, that's right."). The giver's earlier sentence is taken back: recorded "corrected by Meera, and Arjun agreed". Only when the two give that point different values (numbers or products). | `apply_roles` |
| The receiver questions what the giver just said, and nobody answers: "Alerts go to Slack." / "Isn't it PagerDuty now?" / "Let's not go down that path." | The giver's sentence, with a "Questioned in the session, not answered" warning, a "confirm first" row on the incident card and a follow-up line. The question is not a fact. | `apply_roles` (kind "challenged"), `knowledge_builder.apply_conflicts` |
| The receiver states something the giver never answers | Not shown as a fact: "Said by Meera, not confirmed by Arjun: ..." on the follow-up list. | `apply_roles` (kind "unconfirmed") |
| Small talk, asking to repeat, moving the conversation on: "How was the offsite?", "Can you repeat that?", "Let's park that, we're short on time" | Left out, recorded with the reason. Recognised by social phrasing, not by nouns: "our food delivery platform" stays. | `speaker_roles.is_small_talk`, `_REPEAT_RE`, `_DEFLECTS_RE` |

## What counts as a correction signal

- **Same sentence:** "sorry", "I mean", "or rather", "no wait", "correction",
  "scratch that", "my mistake", optionally after "no,", a comma, a dash or a
  pause. A bare "no" or "actually" is not enough ("9 to 5, no 24/7 support";
  "5 minutes, actually 10 on a bad day"). After a dash, or with "new" or a
  word of change after it, "actually" withdraws the value before it: the
  sentence is then kept whole (`withdrawn_values`).
- **Next sentence:** a sentence opening with one of the markers above, "wait",
  "forget what I said" or "I was wrong"; or one that says the old value "is
  old", "is outdated", "is wrong", "use the latter", "the correct value is".
  An opening "actually" counts only with a word of change ("now",
  "upgraded", "moved", "changed"): "Actually, the rollback takes 10
  minutes" adds a fact.
- **Further back:** "I was wrong about X", "forget what I said about X",
  "what I said about X". X must name the earlier sentence's topic. Or the
  earlier statement said again ("I said earlier ...", "Earlier I said ...")
  and then withdrawn ("that's wrong", "that's not right", or a next sentence
  opening with a marker).
- **The values must be of one kind:** the same unit after the number ("7
  days" / "30 days", "2000K" / "3000K") or the same word naming it ("port
  8080" / "port is 8443", "PostgreSQL 14" / "PostgreSQL 16"). Two days of
  the week are of one kind.

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
