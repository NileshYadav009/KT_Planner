# KT receiver feedback: reading a generated PDF as the new owner (3 Nov 2026)

**The setup.** I took the role of a DevOps engineer who has just been given
Quayside, the outgoing owner has left, and the only thing I have is the
generated KT PDF. Quayside is a visible dev KT (`evaluation/dev/d11`): a
port's truck-slot booking system on AWS ECS, handed over as the new owner's
questions and the outgoing owner's answers, about 600 words. It was built
without an LLM, as a laptop with no Groq key would build it, and read twice:
before the fixes of this audit and after them. 13 pages after (14 before):
the cover and contents, the body on pages 3 to 10, the coverage report on 10
to 12, the Sources appendix on 12 and 13.

## Day-1 readiness: twelve questions, answered from the PDF only

| # | Question | In the PDF? | Easy to find? | Detailed enough? | What is missing | Section |
|---|---|---|---|---|---|---|
| 1 | What does the application do? | Yes: the booking system hauliers use to reserve a slot to drop off or collect a container; 9,000 bookings a day; critical, trucks queue when it is down | Yes, first lines of System Overview | Yes | | System Overview |
| 2 | What are its main components? | Yes: Node.js booking API, Python slot allocation worker, React front end on CloudFront, Aurora PostgreSQL, SQS, ElastiCache Redis, ECS on Fargate, Bitbucket, ECR, Secrets Manager | Yes: Technology summary and Architecture Details | Mostly | The Technology summary lists the backend as Node.js only; the Python worker is in the text but not the table | System Overview, Architecture |
| 3 | How does a request travel through the architecture? | Partly: front end from CloudFront, bookings in Aurora, the worker takes jobs from SQS, Redis holds slot availability | Spread over three bullets | No | How the front end reaches the API was not said. The diagram draws 3 of the stated connections; "the worker picks jobs off an SQS queue" and "served from CloudFront" have no line | Architecture |
| 4 | Where do I check if it is down? | Partly: CloudWatch dashboards, Datadog APM on the booking API, the main alert (booking latency over 2 s for 5 min pages through Opsgenie) | Yes: Monitoring, and the card's Alerting row | No | No dashboard name or link, no log location; the KT did not say them, and Monitoring reads "Strong" because its three topics are tools, alerts and paging | Monitoring |
| 5 | How do I identify a failed deployment? | No | | | Not said. The coverage matrix does not ask for it, so its absence is not on the follow-up list | Deployment & Rollback |
| 6 | How do I roll back? | Yes: redeploy the previous task definition revision | Yes: on the card | Enough to start | The console or CLI steps were not said; "Rollback time and approval" is on the follow-up list | Card, Deployment |
| 7 | What are the most common production issues? | Yes: the slot worker falls behind when a ship arrives late (scale the worker service to ten tasks; caught up in about twenty minutes); the customs feed sends duplicate release messages (clear the duplicate in the admin screen) | Yes: on the card | Yes | | Card, Common Failures |
| 8 | What are the critical dashboards? | Partly: "CloudWatch dashboards for the services" | | No | No names, as in 4 | Monitoring |
| 9 | What should I never modify manually? | Yes: never purge the SQS queue to fix a backlog, those are real bookings | Yes: card, Danger Zones | Yes | | Danger Zones |
| 10 | Who owns the infrastructure? | Partly: the terminal systems team owns Quayside | Yes | Mostly | Who owns the three AWS accounts was not said separately | Ownership |
| 11 | How do I escalate a production issue? | Yes: Opsgenie on-call; Grace Okafor, head of terminal systems, if the outgoing owner is not around | Yes: card | Mostly | No phone or channel; the KT gave a name only | Ownership |
| 12 | What should I complete in my first week? | Partly: on day one, join the Opsgenie rota and get read access to production; in the first month, do a production release alone. Schedule a restore test | Day-1 and 30-Day sections, and Disaster Recovery | No | Weeks 1-3 not said; they are on the follow-up list | Day-1, First 30 Days |

**Verdict.** I could keep Quayside running on day one: I know what it does,
what it runs on, how to roll back, how to fix the two failures that happen,
what never to do, and who to call. I could not yet tell a failed release
from a slow one, open the right dashboard, or trace a request end to end.
Those were not said in the session. The document is honest about most of
them on its follow-up list, but not about 5 and 8, because the coverage
topics do not ask for them (`FEATURE_ROADMAP.md`, P2).

## What got in the way, and what was done

| What I saw (before) | Why it mattered | Fixed (after) |
|---|---|---|
| About a dozen facts carried my own question: "Is there a cache? Yes, ElastiCache Redis holds …", "Is there anything I must never do? Never purge the SQS queue …" | Noise in every section; questions in a handover read as doubts | A yes/no answer that is a whole statement stands alone; the question stays only where the answer needs it ("What does it cost? About eleven thousand dollars a month.") |
| "Can we start with what Quayside does? Sure." was the first fact of System Overview | Conversation printed as content | Left out, as a request to talk |
| Disaster Recovery's stated gap was "Backups are nightly, I assume? No, Aurora takes continuous backups …", and it was on the follow-up list | The one thing the outgoing owner answered clearly was listed as unknown | A corrected assumption is a fact; it shows as "Aurora takes continuous backups with point-in-time restore for fourteen days." |
| "Has anyone tested a restore? Not this year." while the matrix said "DR testing: not discussed" | The real gap was hidden | It is the stated gap, on the follow-up list |
| "That's something you should schedule." was not in the document | My first action as owner was dropped as an empty remark | Kept with the sentence before: "Has anyone tested a restore? Not this year. That's something you should schedule." |
| The card's failure row said "Known failure" and gave "You scale the worker service to ten tasks …" with the customs feed's symptom and fix squeezed into the same cell | During an incident I could not tell which fix goes with which failure | One row per failure, each named: "The slot worker falls behind when a ship arrives late … The queue backs up and hauliers see stale slots" → scale to ten tasks; "The customs feed: it sometimes sends duplicate release messages …" → clear the duplicate |
| Tribal Knowledge listed "Is there anything I must…" as safety-critical with "no source" | A safety rule shown as a cut-off question | "Never purge the SQS queue to fix…", sourced |
| The follow-up list asked me to confirm "Thanks for doing this." | Trust in the list drops | Thanks and small talk are not claims; left out |
| Common Failures read "Partial: … 1 only in passing under System Overview" and the follow-up list asked about "Known failures only mentioned in passing" | Two failures with fixes were described; the line said otherwise | Failures said by what goes wrong ("falls behind", "backs up", "stale", "runs out of") count |

## Information in the PDF, by how much the receiver needs it

| Class | What | Recommendation |
|---|---|---|
| Essential | The incident card; Danger Zones; Deployment & Rollback; Common Failures; Ownership & Escalation; Day-1; Disaster Recovery's targets, backups and the untested restore | Keep first, as now |
| Useful | System Overview; Architecture Details and the diagram; Monitoring; Environments; Operational Calendar; Cost; Open Responsibilities; 30-Day plan; the follow-up list | Keep |
| Supporting | The Technology summary; the connections table; the coverage matrix; the Sources appendix (about 3 of 13 pages together) | Move the coverage matrix and Sources to an appendix after Sign-off: they are for checking, not for reading |
| Repetitive or empty | Tribal Knowledge rows that repeat Danger Zones word for word; Architecture Metadata (three rows of "Not discussed"); "Not Covered in This KT" (the same as the matrix's Missing rows); the Sign-off placeholder before any sign-off; questions kept with answers that lean on them | Show Tribal Knowledge rows only when they add something; fold empty metadata rows and "Not Covered" into the follow-up list; drop the placeholder until a sign-off starts |

The appendix and folding changes are recommendations, not done: they
change what the release gates and the matrix read on the page, so they need
their own measurement.

## Notes on accuracy

- Nothing in the document was invented: every line is the speaker's
  sentence with its source; "Alerting (no first step stated)" says so instead
  of guessing a step.
- The questions still printed are those whose answers need them; rewriting
  "About eleven thousand dollars a month" into a sentence needs an LLM, and
  with Groq the polish does it under its checks.
- One reading of one KT. The fixes are generic and were measured on every
  visible KT (`docs/REPOSITORY_AUDIT.md` §9abj); other KT styles may show
  other problems, which is why matrix case T04 now tests these patterns in
  words that are not Quayside's.
