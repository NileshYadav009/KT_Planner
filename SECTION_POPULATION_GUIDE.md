# Section Population Guide: What Content Each Section Needs

## Overview

The KT document has 13 required sections. Each section needs at least 2 sentences to move from "weak" to "covered" status. This guide explains what type of content each section should contain and how to recognize it in your transcript.

---

## 1. SYSTEM OVERVIEW ⭐ [REQUIRED]

**Purpose**: Give the incoming owner a 5-minute understanding of what this system is and why it matters.

**Keywords to look for in transcript**:
- "The system handles...", "This platform does..."
- "Our main product is...", "We built this to..."
- "Customers use this for...", "Internal teams depend on..."
- "If it goes down, we lose...", "Business criticality is..."
- "It processes...", "It manages...", "It stores..."

**Content template**:
```
Sentence 1: "System X is a [platform/service] that handles [main purpose]."
Sentence 2: "It's used by [B2B/B2C/Internal] customers for [use case]."
Sentence 3: "If it goes down, [business impact] is at risk."
```

**Example**:
- ✅ "Our payments platform processes transactions for 10,000 merchants."
- ✅ "We handle $500M in GMV annually and are critical to revenue."
- ❌ "We deploy using Jenkins" (this is deployment info, not system overview)

---

## 2. ARCHITECTURE REFERENCE ⭐ [REQUIRED]

**Purpose**: Point to where the architecture is documented and confirm it's current.

**Keywords to look for**:
- "Architecture diagram in...", "See the docs at..."
- "Figma link:", "Confluence page:", "GitHub wiki:"
- "Last updated...", "Verified on..."
- "You can find the architecture in...", "Diagram is on..."

**Content template**:
```
Sentence 1: "Architecture documentation lives at [URL]."
Sentence 2: "Last verified on [date] by [person]."
Sentence 3: "[Brief description of main components: database, services, etc.]"
```

**Example**:
- ✅ "Architecture diagrams are in our Confluence: https://confluence.company.com/payments-arch. Last verified 2025-Q4 by Sarah Chen."
- ✅ "The system uses microservices: API Gateway → Payment Service → Database → Cache."
- ❌ "We have three servers and they run on AWS" (too vague, not architectural reference)

---

## 3. PLAIN-ENGLISH NOTES [Optional but valuable]

**Purpose**: Tribal knowledge, shortcuts, gotchas, and learned lessons.

**Keywords to look for**:
- "Beware of...", "Watch out for...", "Don't forget about..."
- "Pro tip:", "Shortcut:", "Gotcha:"
- "This is weird because...", "Nobody mentioned but..."
- "The trick is...", "Lessons learned..."
- "Sharp edge:", "History: this used to..."

**Content template**:
```
Sentence 1: "Gotcha: [warning about unusual behavior or hidden complexity]"
Sentence 2: "Pro tip: [shortcut or workaround discovered in production]"
Sentence 3: "History: [why something is the way it is]"
```

**Example**:
- ✅ "Beware: queue sometimes appears full but isn't. Check Redis directly."
- ✅ "Pro-tip: restart the cache layer first, then the API layer, not vice versa."
- ✅ "History: we migrated from MySQL to Postgres in 2023, old scripts still reference MySQL."

---

## 4. DAY-1 SURVIVAL CHECKLIST ⭐ [REQUIRED]

**Purpose**: By end of day 1, what access and knowledge does the new owner need?

**Keywords to look for**:
- "First day:", "Day one:", "As soon as you get access..."
- "Tools you need:", "Access to...", "Passwords in..."
- "Read-only checks:", "Safe to explore:", "Don't touch!"
- "Monitoring dashboard:", "Pipeline view:", "Status page..."
- "Required access: AWS, GitHub, Slack, PagerDuty, Datadog..."

**Content template**:
```
Sentence 1: "Day-1 access needed: [list of systems: AWS, GitHub, Datadog, Slack channels]"
Sentence 2: "Safe first actions: Review [monitoring dashboard], Check [CI/CD pipeline], Read [documentation]"
Sentence 3: "DO NOT touch [danger zones] on day 1."
```

**Example**:
- ✅ "Access: AWS prod account, GitHub payments-core repo, PagerDuty on-call rotation, Slack #payments-eng."
- ✅ "Safe first day: Review Datadog dashboard to see system health, check recent deployments in Jenkins, read runbook."
- ✅ "Do not: run any database migrations or edit kubernetes configs without pair programming first."

---

## 5. DEPLOYMENT & ROLLBACK ⭐ [REQUIRED]

**Purpose**: Exact steps to deploy code to production and how to undo it.

**Keywords to look for**:
- "Deploy to prod:", "Deployment process:", "Steps to deploy..."
- "Pipeline:", "Trigger:", "Manual vs automatic"
- "Pre-checks:", "Post-deployment:", "Validation:"
- "Rollback:", "Undo:", "How to revert..."
- "Deployment window:", "Can we deploy on Friday?"
- "Who approves?", "How long does it take?"

**Content template**:
```
Sentence 1: "Normal deployment: [Step 1] → [Step 2] → [Step 3]"
Sentence 2: "Process: Create PR → Get approval → Merge → CI/CD runs → [result]"
Sentence 3: "Rollback: If issues occur within 1 hour, run [rollback command]. If 1+ hours, use backup restore."
```

**Example**:
- ✅ "Deploy: Push to main → CI tests run → Docker image builds → Kubernetes deployment auto-triggers → system live in 3 minutes"
- ✅ "Pre-checks: Verify all tests pass, check no open alerts"
- ✅ "Rollback: Keep last 3 images in prod. Use kubectl rollout undo for instant v-1 (takes 30 seconds)"

---

## 6. COMMON FAILURES & FIXES ⭐ [REQUIRED]

**Purpose**: What breaks, how to detect it, and what to do.

**Keywords to look for**:
- "Common failure:", "When X happens:", "If you see..."
- "Symptoms:", "Root cause:", "How to fix:"
- "Known issue:", "We've seen it happen when..."
- "Happens every...", "Happens during...", "Triggered by..."
- "Debug steps:", "Solution:", "Workaround:"

**Content template**:
```
Sentence 1: "Failure: [symptom]. Cause: [root cause]. Fix: [action]"
Sentence 2: "Failure: [another common issue]. Symptoms: [what you see]. Resolution: [steps]"
Sentence 3: "If nothing works: [escalation or nuclear options]"
```

**Example**:
- ✅ "Database connection pool exhaustion: Connections hang, timeouts spike. Fix: Restart connection pool via Datadog [link]."
- ✅ "Memory leak in old worker: After 2 days uptime, memory > 90%. Fix: Auto-scale policy kicks in, but manually restart if urgent."
- ✅ "If fixing doesn't work: Page on-call lead (defined in PagerDuty #1 escalation policy)."

---

## 7. KNOWN BAD DAYS / WINDOWS ⭐ [REQUIRED]

**Purpose**: When is the system acting weird? What times should we avoid deploying?

**Keywords to look for**:
- "Friday deployments:", "Never deploy on...", "Bad time to deploy:"
- "Maintenance window:", "Scheduled downtime:", "Planned maintenance..."
- "Known issue on...", "Acts up on...", "Happens during..."
- "Weekends:", "Month-end:", "Year-end:", "During reports..."
- "Budget reconciliation...", "Tax season...", "Black Friday..."

**Content template**:
```
Sentence 1: "Do NOT deploy on: [days/times]"
Sentence 2: "Reason: [business event]. Dates: [specific dates]. Duration: [timing]"
Sentence 3: "Alternative window: [when is safe to deploy]. Approval needed from: [person/team]"
```

**Example**:
- ✅ "Do NOT deploy on Fridays after 3 PM. Reason: weekend coverage is limited, can't respond to incidents."
- ✅ "Month-end (last 3 days): Reconciliation team running complex queries, extra load on database. Avoid deploys."
- ✅ "Black Friday: We disable auto-scaling thresholds and freeze code changes. Pre-approved changes only."

---

## 8. DANGER ZONES (DO NOT TOUCH) ⭐ [REQUIRED]

**Purpose**: What operations are risky and could cause system outage?

**Keywords to look for**:
- "Never run:", "Do not execute:", "Dangerous:"
- "Will cause outage:", "Will delete data:", "Irreversible:"
- "Requires manual review:", "High-risk operation:", "Ask before..."
- "Requires approval from:", "Only run if...", "Must coordinate with..."

**Content template**:
```
Sentence 1: "DANGER: Do not run [specific command/operation]. Why: [impact]"
Sentence 2: "DANGER: Do not touch [configuration/system]. Why: [will break]"
Sentence 3: "If you need to do this: [who to ask], [approval process], [safer way to do it]"
```

**Example**:
- ✅ "DANGER: Do not run 'DROP TABLE transactions'. It will delete production data. No backup exists for < 24 hours ago."
- ✅ "DANGER: Do not restart the cache layer without notifying the frontend team. It causes 10-min outage."
- ✅ "If DB migration is needed: File a ticket with [DBAdmin team], wait for approval, run during maintenance window."

---

## 9. OWNERSHIP & ESCALATION ⭐ [REQUIRED]

**Purpose**: Who owns what? Who do I call if I'm stuck?

**Keywords to look for**:
- "I own X, they own Y", "Owned by:", "Responsible for:"
- "For questions about..., contact...", "Escalate to:", "Page this person:"
- "On-call rotation:", "PagerDuty:", "Slack channel:", "Email list:"
- "If urgent:", "In a crisis:", "Chain of command:"
- "Cross-team:", "Depends on:", "Coordination needed:"

**Content template**:
```
Sentence 1: "[Service/Component] is owned by [team/person]. [Service 2] is owned by [team/person]."
Sentence 2: "For urgent issues: Page on-call via [PagerDuty]. For non-urgent: Slack [channel]."
Sentence 3: "Escalation path: Me → [team lead] → [director] → VP Eng"
```

**Example**:
- ✅ "DevOps team owns infrastructure. Frontend team owns web UI. Payments team owns business logic."
- ✅ "For incidents: Page payment-oncall via PagerDuty. For questions: Post in #payments-eng Slack."
- ✅ "If I'm blocked: Talk to Jane (payments TL) → Mike (Eng Manager) → CTO"

---

## 10. FIRST 30-DAY OWNERSHIP PLAN ⭐ [REQUIRED]

**Purpose**: What should the new owner accomplish in their first month?

**Keywords to look for**:
- "Week 1:", "Week 2-4:", "First month goals:"
- "You should:", "Focus on:", "Learn about:", "Document..."
- "30 days:", "By end of month:", "Ramp-up plan:"
- "Milestones:", "Checkpoints:", "Deliverables:"
- "What you'll know by day 30:"

**Content template**:
```
Sentence 1: "Week 1: [initial setup tasks]. Week 2-3: [learning/documentation]. Week 4: [take-over tasks]"
Sentence 2: "By day 30: You should be able to [specific capability], understand [key concept], handle [type of incident]"
Sentence 3: "Checkpoints: Day 5 [milestone], Day 15 [milestone], Day 30 [milestone]"
```

**Example**:
- ✅ "Week 1: Environment setup, read architecture, join on-call rotation. Week 2-3: Pair on deployments, debug prod issues. Week 4: Own patches independently."
- ✅ "By day 30: Deploy code solo, respond to P2 incidents, understand request flow from UI to database."
- ✅ "Checkpoints: Day 5-deploy something, Day 15-handle incident, Day 30-own sprint independently."

---

## 11. OPEN RESPONSIBILITIES & TRANSITION PLAN ⭐ [REQUIRED]

**Purpose**: Is the transition complete? What tasks are still hanging?

**Keywords to look for**:
- "Still need to:", "Pending:", "In progress:", "TODO:"
- "Not done yet:", "Blockers:", "Waiting on..."
- "Handover incomplete:", "Partially migrated:", "Still using old..."
- "Deadline:", "Target completion:", "ETA:"
- "My responsibility:", "Your responsibility:", "Shared..."

**Content template**:
```
Sentence 1: "Pending: [task] - owner: [person], due: [date]"
Sentence 2: "Incomplete: [transition item]. Status: [current status]. Blocker: [if any]"
Sentence 3: "Transition complete by: [date]. Fallback contact: [person] until then."
```

**Example**:
- ✅ "Migrate from old database - Jane owns this, due March 31. Blocks: waiting on schema review."
- ✅ "Update documentation - I will do this by end of week. Create runbook - postponed to Q2."
- ✅ "Full transition complete by April 15. Until then, reach out to outgoing owner Sarah if critical."

---

## 12. HANDOVER COMPLETION CHECK ⭐ [REQUIRED]

**Purpose**: Verification that everything is handed over and documented.

**Keywords to look for**:
- "Verified:", "Confirmed:", "Signed off:"
- "Checklist:", "All items complete:", "Nothing pending:"
- "Final check by:", "Approved by:", "Verified by:"
- "Date:", "Signature:", "Sign-off:"

**Content template**:
```
Sentence 1: "✓ Incoming owner has reviewed [list]. ✓ Outgoing owner verified [list]. ✓ No pending transitions."
Sentence 2: "Knowledge transfer complete. System is production-ready for new ownership."
Sentence 3: "Verified on [date] by [incoming owner]. Approved by [outgoing owner] / [manager]"
```

**Example**:
- ✅ "✓ Access verified, ✓ Docs reviewed, ✓ Incident handled solo, ✓ Architecture understood, ✓ Deployments independent, ✓ No blockers remain."
- ✅ "Transition complete and verified. Bob ready for full ownership."
- ✅ "Verified 2026-03-30 by Bob Chen (new owner). Approved by Alice Wang (previous owner)."

---

## 13. Sign-off ⭐ [REQUIRED]

**Purpose**: Official sign-off from involved parties.

**Keywords to look for**:
- "Approved by:", "Signed by:", "Sign-off from:"
- "Manager:", "Director:", "Both owners:"
- "Date:", "Signature:", "Confirmation:"
- "Verified by:", "Acknowledged by:"

**Content template**:
```
Sentence 1: "Incoming owner ([name]) confirms readiness to take over. Date: [date]. Signature: [confirmation]"
Sentence 2: "Outgoing owner ([name]) confirms transition complete. Date: [date]. Signature: [confirmation]"
Sentence 3: "Manager ([name]) approves ownership transfer. Date: [date]"
```

**Example**:
- ✅ "Incoming owner (Bob Chen) confirms: Ready to take over. 2026-03-30."
- ✅ "Outgoing owner (Alice Wang) confirms: All knowledge transferred. 2026-03-30."
- ✅ "Manager (Mike Rodriguez) approves handover. 2026-03-30."

---

## How to Use This Guide

1. **Check your transcript** for keywords matching each section
2. **Use `/diagnose`** to see unassigned sentences
3. **Use `/populate-section`** to auto-assign matching sentences
4. **Manually edit** any sections that need custom content
5. **Verify coverage** reaches 100% with `/coverage` endpoint

Each section needs 2+ sentences to move to "covered" status. Use the templates above as guides for formatting.
