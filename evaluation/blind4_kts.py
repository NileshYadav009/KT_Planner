"""Round-8 blind KTs (written 26 Oct 2026, before any result on them).

    python blind4_kts.py OUTDIR     # writes one golden JSON per KT, checks quotes

Sealed for the accuracy P0 work (audit §9abd): the classifier designs are
chosen on the visible KTs only, and this set is read once, at the end, in
aggregate (`accuracy.py LABEL blind4`). Nothing here is ever printed or
tuned on one fact at a time. Written by the same author as the fixes, so not
independent of them.

Four styles the earlier sets lack: a two-person conversation in which the
receiver guesses wrong and the giver corrects it, terse spoken notes for an
on-premises Windows system with self-corrections, a screen-share walkthrough,
and a rambling, hedged KT that leaves three sections undiscussed. Annotation
follows the dev KTs' conventions.
"""
import json
import os
import re
import sys

S = "written for round 8 (synthetic, sealed)"


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(t).lower())).strip()


AR = ["architecture_reference", "system_overview"]
DAY1 = ["day1_survival_checklist", "first_30_day_ownership"]
MONTH1 = ["first_30_day_ownership", "handover_completion"]
ALL15 = ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
         "common_failures", "disaster_recovery", "security_controls", "danger_zones", "known_bad_days", "cost_optimization",
         "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"]
KTS = []

# ---------------------------------------------------------------------------
KTS.append({
    "id": "b08_routemaster_ecs_conversation",
    "title": "Routemaster: ECS dispatch, giver and receiver conversation",
    "proves": "Two speakers; the receiver's wrong guess is corrected; answers carry the facts.",
    "transcript": """Tom: Thanks for making time, Aisha. This is the handover for Routemaster before I move to the payments team.
Aisha: Sure. I've used the driver app but never looked at the backend.
Tom: Routemaster assigns delivery jobs to drivers for our three warehouses. Dispatchers use it all day, and the driver app polls it for the next stop. On a normal day it plans around eighteen thousand stops.
Aisha: Where does it run?
Tom: It's on Amazon ECS with Fargate. The dispatch API is a Node.js service, and route optimisation runs as a separate Python worker.
Aisha: So the jobs are stored in Postgres, right?
Tom: No, the jobs table is in DynamoDB. Postgres was the old system before the rewrite.
Aisha: Got it. And how do the workers get the work?
Tom: The API puts each planning request on an SQS queue, and the optimiser workers pull from that queue. Driver locations come in through API Gateway and a Lambda function that writes them to DynamoDB.
Aisha: How do you release?
Tom: GitHub Actions builds the container image and pushes it to ECR, then the pipeline updates the ECS service with a rolling deployment. If a release goes bad, redeploy the previous task definition revision from the ECS console. Releases to production need a second engineer to approve the pull request.
Aisha: How many environments?
Tom: Staging and production. Staging uses a copy of last month's warehouse data, so you can test route plans there safely.
Aisha: And monitoring?
Tom: CloudWatch dashboards show queue depth and API latency. There's an alarm when the SQS queue has more than five hundred messages waiting, and it pages whoever is on call through Opsgenie.
Aisha: What breaks?
Tom: The usual one is the optimiser running out of memory on very large routes, and then the queue backs up. Increase the task memory to eight gigabytes and restart the workers. The other is DynamoDB throttling when the morning planning starts; we switched that table to on-demand capacity and it has mostly stopped.
Aisha: What about backups?
Tom: Point-in-time recovery is turned on for the DynamoDB tables, which gives us thirty five days. We can recover the service within two hours, but we've never practised a full restore.
Aisha: Security?
Tom: Each ECS task has its own IAM role, and the Google Maps API key lives in Secrets Manager. Drivers sign in through Cognito. One warning: never purge the SQS queue during the day, because the dispatchers lose every plan that's waiting.
Aisha: When is it busiest?
Tom: The week before Christmas is the worst, volumes double. And Monday mornings are heavy because weekend orders all land at once.
Aisha: What does it cost?
Tom: Around nine thousand dollars a month, mostly Fargate. We moved the optimiser workers to Fargate Spot, which cut that part by about half.
Aisha: Who owns it?
Tom: The logistics platform team owns Routemaster. If something is on fire, escalate to Rachel Kim, the logistics engineering manager. Something unfinished: the move to the new traffic data provider is half done. The contract is signed but the integration hasn't started.
Aisha: What should I do first?
Tom: On day one, get access to the AWS production account and join the Opsgenie rotation. In the first month, plan and run one release yourself.""",
    "facts": [
        {"id": "F01", "quote": "Routemaster assigns delivery jobs to drivers for our three warehouses", "in": ["system_overview"]},
        {"id": "F02", "quote": "Dispatchers use it all day", "in": ["system_overview"]},
        {"id": "F03", "quote": "the driver app polls it for the next stop", "in": AR},
        {"id": "F04", "quote": "On a normal day it plans around eighteen thousand stops", "in": ["system_overview"]},
        {"id": "F05", "quote": "It's on Amazon ECS with Fargate", "in": AR},
        {"id": "F06", "quote": "The dispatch API is a Node.js service", "in": AR},
        {"id": "F07", "quote": "route optimisation runs as a separate Python worker", "in": AR},
        {"id": "F08", "quote": "the jobs table is in DynamoDB", "in": AR},
        {"id": "F09", "quote": "The API puts each planning request on an SQS queue", "in": AR},
        {"id": "F10", "quote": "the optimiser workers pull from that queue", "in": AR},
        {"id": "F11", "quote": "Driver locations come in through API Gateway and a Lambda function that writes them to DynamoDB", "in": AR},
        {"id": "F12", "quote": "GitHub Actions builds the container image and pushes it to ECR", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "the pipeline updates the ECS service with a rolling deployment", "in": ["deployment_and_rollback"]},
        {"id": "F14", "quote": "redeploy the previous task definition revision from the ECS console", "in": ["deployment_and_rollback"]},
        {"id": "F15", "quote": "Releases to production need a second engineer to approve the pull request", "in": ["deployment_and_rollback"]},
        {"id": "F16", "quote": "Staging and production", "in": ["environments"]},
        {"id": "F17", "quote": "Staging uses a copy of last month's warehouse data", "in": ["environments"]},
        {"id": "F18", "quote": "CloudWatch dashboards show queue depth and API latency", "in": ["monitoring_observability"]},
        {"id": "F19", "quote": "There's an alarm when the SQS queue has more than five hundred messages waiting", "in": ["monitoring_observability"]},
        {"id": "F20", "quote": "it pages whoever is on call through Opsgenie", "in": ["monitoring_observability", "ownership_escalation"]},
        {"id": "F21", "quote": "the optimiser running out of memory on very large routes", "in": ["common_failures"]},
        {"id": "F22", "quote": "Increase the task memory to eight gigabytes and restart the workers", "in": ["common_failures"]},
        {"id": "F23", "quote": "DynamoDB throttling when the morning planning starts", "in": ["common_failures"]},
        {"id": "F24", "quote": "we switched that table to on-demand capacity", "in": ["common_failures"]},
        {"id": "F25", "quote": "Point-in-time recovery is turned on for the DynamoDB tables", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "We can recover the service within two hours", "alt": ["two hours"], "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "we've never practised a full restore", "in": ["disaster_recovery"]},
        {"id": "F28", "quote": "Each ECS task has its own IAM role", "in": ["security_controls"]},
        {"id": "F29", "quote": "the Google Maps API key lives in Secrets Manager", "in": ["security_controls"]},
        {"id": "F30", "quote": "Drivers sign in through Cognito", "in": ["security_controls"]},
        {"id": "F31", "quote": "never purge the SQS queue during the day", "in": ["danger_zones"]},
        {"id": "F32", "quote": "The week before Christmas is the worst, volumes double", "in": ["known_bad_days"]},
        {"id": "F33", "quote": "Monday mornings are heavy because weekend orders all land at once", "in": ["known_bad_days"]},
        {"id": "F34", "quote": "Around nine thousand dollars a month, mostly Fargate", "in": ["cost_optimization"]},
        {"id": "F35", "quote": "We moved the optimiser workers to Fargate Spot", "in": ["cost_optimization"]},
        {"id": "F36", "quote": "The logistics platform team owns Routemaster", "in": ["ownership_escalation"]},
        {"id": "F37", "quote": "escalate to Rachel Kim, the logistics engineering manager", "in": ["ownership_escalation"]},
        {"id": "F38", "quote": "the move to the new traffic data provider is half done", "in": ["open_responsibilities"]},
        {"id": "F39", "quote": "get access to the AWS production account and join the Opsgenie rotation", "in": DAY1},
        {"id": "F40", "quote": "plan and run one release yourself", "in": MONTH1},
    ],
    "covered": ALL15,
    "missing": [],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "b09_dispensa_onprem_terse",
    "title": "Dispensa: on-premises pharmacy system, terse spoken notes",
    "proves": "Fragments without verbs, labels said aloud, self-corrections, a section not discussed.",
    "transcript": """Right, quick handover for Dispensa before I go on leave. Dispensa is the pharmacy dispensing system for the four hospitals in the trust. Pharmacists check every prescription in it before anything leaves the pharmacy. Busy system, roughly six thousand prescriptions a day.

Stack. Front end is an ASP.NET web app on IIS, two web servers behind an F5 load balancer. Database is SQL Server 2019 in an Always On availability group, two nodes in the main data centre. The robot dispensers talk to it through a Windows service called RoboLink. Orders arrive from the electronic patient record over HL7 through the Rhapsody integration engine.

Environments: test, training and live. Training is refreshed from live every Sunday with patient names scrambled.

Releases. We use Octopus Deploy. Each release goes to test, then training, then live. Live releases happen on Tuesday evenings after eight, never during pharmacy opening hours. Rollback is Octopus redeploying the previous release; the database scripts have to be reversed by hand, so check with the DBA first.

Monitoring: SCOM watches the servers and the RoboLink service. If RoboLink stops, SCOM emails the service desk. There's also a SQL Agent job that checks the HL7 queue every ten minutes, sorry, every five minutes, and raises an alert if more than fifty messages are stuck.

Problems. Number one: RoboLink loses its connection to a robot after a network blip and the robot stops picking. Restart the RoboLink service on that hospital's server. Number two: HL7 messages stuck because of an invalid ward code. Fix the ward code in the mapping table in Rhapsody and resend the messages.

Backups: full backup nightly, log backups every fifteen minutes, kept for thirty days. Recovery point objective fifteen minutes, recovery time objective four hours. The secondary node is in the same building, so a site loss would take the whole service down.

Security: users log in with their NHS smartcard. Controlled drug records need two pharmacists to sign. Audit tables cannot be edited by anyone, including us.

Do not restart the SQL Server primary during the day; a failover drops every pharmacist's session and the robots stop.

Winter is the worst period, flu season doubles the discharges. Also the first Wednesday of August, when the new junior doctors start.

Owned by the clinical applications team. Out of hours, call the on-call clinical apps engineer through the switchboard. Escalation after that is James Whitfield, head of clinical systems.

Still open: the upgrade to SQL Server 2022, planned but not scheduled. And the supplier support contract renewal is due in March.

First day: get your smartcard set up and admin rights on the test servers. First month: do one Tuesday release with me watching, sorry, with Kevin watching since I'll be away.""",
    "facts": [
        {"id": "F01", "quote": "Dispensa is the pharmacy dispensing system for the four hospitals in the trust", "in": ["system_overview"]},
        {"id": "F02", "quote": "Pharmacists check every prescription in it before anything leaves the pharmacy", "in": ["system_overview"]},
        {"id": "F03", "quote": "roughly six thousand prescriptions a day", "in": ["system_overview"]},
        {"id": "F04", "quote": "Front end is an ASP.NET web app on IIS", "alt": ["Frontend is an ASP.NET web app on IIS"], "in": AR},
        {"id": "F05", "quote": "two web servers behind an F5 load balancer", "in": AR},
        {"id": "F06", "quote": "Database is SQL Server 2019 in an Always On availability group", "in": AR},
        {"id": "F07", "quote": "The robot dispensers talk to it through a Windows service called RoboLink", "in": AR},
        {"id": "F08", "quote": "Orders arrive from the electronic patient record over HL7 through the Rhapsody integration engine", "in": AR},
        {"id": "F09", "quote": "test, training and live", "in": ["environments"]},
        {"id": "F10", "quote": "Training is refreshed from live every Sunday with patient names scrambled", "in": ["environments"]},
        {"id": "F11", "quote": "We use Octopus Deploy", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "Each release goes to test, then training, then live", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "Live releases happen on Tuesday evenings after eight", "in": ["deployment_and_rollback", "known_bad_days"]},
        {"id": "F14", "quote": "never during pharmacy opening hours", "in": ["deployment_and_rollback", "known_bad_days"]},
        {"id": "F15", "quote": "Rollback is Octopus redeploying the previous release", "in": ["deployment_and_rollback"]},
        {"id": "F16", "quote": "the database scripts have to be reversed by hand", "in": ["deployment_and_rollback"]},
        {"id": "F17", "quote": "SCOM watches the servers and the RoboLink service", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "If RoboLink stops, SCOM emails the service desk", "in": ["monitoring_observability"]},
        {"id": "F19", "quote": "raises an alert if more than fifty messages are stuck", "in": ["monitoring_observability"]},
        {"id": "F20", "quote": "RoboLink loses its connection to a robot after a network blip", "in": ["common_failures"]},
        {"id": "F21", "quote": "Restart the RoboLink service on that hospital's server", "in": ["common_failures"]},
        {"id": "F22", "quote": "HL7 messages stuck because of an invalid ward code", "in": ["common_failures"]},
        {"id": "F23", "quote": "Fix the ward code in the mapping table in Rhapsody and resend the messages", "in": ["common_failures"]},
        {"id": "F24", "quote": "full backup nightly", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "log backups every fifteen minutes", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "Recovery point objective fifteen minutes", "alt": ["fifteen minutes"], "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "recovery time objective four hours", "alt": ["four hours"], "in": ["disaster_recovery"]},
        {"id": "F28", "quote": "The secondary node is in the same building", "in": ["disaster_recovery"]},
        {"id": "F29", "quote": "users log in with their NHS smartcard", "in": ["security_controls"]},
        {"id": "F30", "quote": "Controlled drug records need two pharmacists to sign", "in": ["security_controls"]},
        {"id": "F31", "quote": "Audit tables cannot be edited by anyone", "in": ["security_controls"]},
        {"id": "F32", "quote": "Do not restart the SQL Server primary during the day", "in": ["danger_zones"]},
        {"id": "F33", "quote": "Winter is the worst period, flu season doubles the discharges", "in": ["known_bad_days"]},
        {"id": "F34", "quote": "the first Wednesday of August, when the new junior doctors start", "in": ["known_bad_days"]},
        {"id": "F35", "quote": "Owned by the clinical applications team", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "call the on-call clinical apps engineer through the switchboard", "in": ["ownership_escalation"]},
        {"id": "F37", "quote": "James Whitfield, head of clinical systems", "in": ["ownership_escalation"]},
        {"id": "F38", "quote": "the upgrade to SQL Server 2022, planned but not scheduled", "in": ["open_responsibilities"]},
        {"id": "F39", "quote": "the supplier support contract renewal is due in March", "in": ["open_responsibilities"]},
        {"id": "F40", "quote": "get your smartcard set up and admin rights on the test servers", "in": DAY1},
        {"id": "F41", "quote": "do one Tuesday release", "in": MONTH1},
    ],
    "covered": [s for s in ALL15 if s != "cost_optimization"],
    "missing": ["cost_optimization"],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "b10_permitly_cloudrun_screenshare",
    "title": "Permitly: Cloud Run permits, screen-share walkthrough",
    "proves": "Narration of what is on screen ('on the left', 'down here'), facts said while pointing.",
    "transcript": """Okay, can everyone see my screen? Great. So this is Permitly, the parking permit system the city council uses. Residents apply for permits online, and the parking wardens check permits from their handheld devices. We have about ninety thousand active permits.

Let me open the architecture diagram. On the left is the public website, which is a React app hosted on Firebase Hosting. In the middle is the API on Cloud Run. Applications are stored in Firestore, and uploaded documents like proof of address go into a Cloud Storage bucket. Payments go through Stripe. When a payment succeeds, Stripe calls our webhook, and the webhook publishes a message to Pub/Sub that the permit issuing service picks up.

Now I'll switch to the Cloud Build page. Every merge to main triggers a build, and the new revision is deployed to Cloud Run with no traffic. Then we move traffic over gradually, ten percent first. If the new revision misbehaves, you shift all traffic back to the previous revision on this screen here. That takes seconds.

This is the projects list. We've got permitly-dev and permitly-prod. Dev uses Stripe's test mode, so no real money moves.

Here's the monitoring dashboard. These charts are request count, error rate and latency for the API. Down here is the Pub/Sub backlog. We have an uptime check on the public site every minute, and alerts go to the team's Google Chat space.

So what goes wrong. You can see this spike in errors here; that was Stripe webhooks failing because the signing secret was rotated and not updated in our config. If that happens, update the secret in Secret Manager and redeploy. The other thing is permit renewals in January: the backlog in Pub/Sub grows and permits are issued late. Raise the maximum instances of the issuing service and it catches up.

Firestore has daily exports to a separate bucket, kept for ninety days. I'm honestly not sure anyone has tried restoring one.

Security. The wardens' devices authenticate with a client certificate. Only two people have the owner role on the production project. All personal data stays in the europe-west2 region.

Please never change the Firestore indexes by hand in production. They are managed in the repo, and a manual change gets overwritten and can break queries.

January is our peak because most permits renew at the start of the year. Also the week after a new parking zone is announced.

Costs are small, about six hundred pounds a month. The biggest item is Cloud Run, and we set minimum instances to zero on dev.

The digital services team owns Permitly. Escalation goes to Amara Osei, the digital services lead.

We still have to switch off the old paper permit process, that's due in the spring. Also the wardens' app needs an update for the new Android version, and nobody has picked it up yet.

For your first day, ask for the Viewer role on prod and Editor on dev. In the first month, try to handle the January renewal peak with someone shadowing you.""",
    "facts": [
        {"id": "F01", "quote": "Permitly, the parking permit system the city council uses", "in": ["system_overview"]},
        {"id": "F02", "quote": "Residents apply for permits online", "in": ["system_overview"]},
        {"id": "F03", "quote": "the parking wardens check permits from their handheld devices", "in": ["system_overview"]},
        {"id": "F04", "quote": "We have about ninety thousand active permits", "in": ["system_overview"]},
        {"id": "F05", "quote": "the public website, which is a React app hosted on Firebase Hosting", "in": AR},
        {"id": "F06", "quote": "the API on Cloud Run", "in": AR},
        {"id": "F07", "quote": "Applications are stored in Firestore", "in": AR},
        {"id": "F08", "quote": "uploaded documents like proof of address go into a Cloud Storage bucket", "in": AR},
        {"id": "F09", "quote": "Payments go through Stripe", "in": AR},
        {"id": "F10", "quote": "the webhook publishes a message to Pub/Sub that the permit issuing service picks up", "in": AR},
        {"id": "F11", "quote": "Every merge to main triggers a build", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "the new revision is deployed to Cloud Run with no traffic", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "Then we move traffic over gradually, ten percent first", "in": ["deployment_and_rollback"]},
        {"id": "F14", "quote": "you shift all traffic back to the previous revision", "in": ["deployment_and_rollback"]},
        {"id": "F15", "quote": "We've got permitly-dev and permitly-prod", "in": ["environments"]},
        {"id": "F16", "quote": "Dev uses Stripe's test mode, so no real money moves", "in": ["environments"]},
        {"id": "F17", "quote": "These charts are request count, error rate and latency for the API", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "Down here is the Pub/Sub backlog", "in": ["monitoring_observability"]},
        {"id": "F19", "quote": "We have an uptime check on the public site every minute", "in": ["monitoring_observability"]},
        {"id": "F20", "quote": "alerts go to the team's Google Chat space", "in": ["monitoring_observability"]},
        {"id": "F21", "quote": "Stripe webhooks failing because the signing secret was rotated and not updated in our config", "in": ["common_failures"]},
        {"id": "F22", "quote": "update the secret in Secret Manager and redeploy", "in": ["common_failures"]},
        {"id": "F23", "quote": "the backlog in Pub/Sub grows and permits are issued late", "in": ["common_failures", "known_bad_days"]},
        {"id": "F24", "quote": "Raise the maximum instances of the issuing service and it catches up", "in": ["common_failures"]},
        {"id": "F25", "quote": "Firestore has daily exports to a separate bucket, kept for ninety days", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "I'm honestly not sure anyone has tried restoring one", "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "The wardens' devices authenticate with a client certificate", "in": ["security_controls"]},
        {"id": "F28", "quote": "Only two people have the owner role on the production project", "in": ["security_controls"]},
        {"id": "F29", "quote": "All personal data stays in the europe-west2 region", "in": ["security_controls"]},
        {"id": "F30", "quote": "never change the Firestore indexes by hand in production", "in": ["danger_zones"]},
        {"id": "F31", "quote": "January is our peak because most permits renew at the start of the year", "in": ["known_bad_days"]},
        {"id": "F32", "quote": "the week after a new parking zone is announced", "in": ["known_bad_days"]},
        {"id": "F33", "quote": "about six hundred pounds a month", "in": ["cost_optimization"]},
        {"id": "F34", "quote": "we set minimum instances to zero on dev", "in": ["cost_optimization"]},
        {"id": "F35", "quote": "The digital services team owns Permitly", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "Escalation goes to Amara Osei, the digital services lead", "in": ["ownership_escalation"]},
        {"id": "F37", "quote": "switch off the old paper permit process, that's due in the spring", "in": ["open_responsibilities"]},
        {"id": "F38", "quote": "the wardens' app needs an update for the new Android version", "in": ["open_responsibilities"]},
        {"id": "F39", "quote": "ask for the Viewer role on prod and Editor on dev", "in": DAY1},
        {"id": "F40", "quote": "handle the January renewal peak with someone shadowing you", "in": MONTH1},
    ],
    "covered": ALL15,
    "missing": [],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "b11_gridmeter_openshift_ramble",
    "title": "Gridmeter: Kafka and OpenShift on premises, rambling and hedged",
    "proves": "Returns to a topic, hedges, a stated gap, three sections never discussed.",
    "transcript": """So, Gridmeter. Where do I start. It's the thing that takes the smart meter readings from about two million homes and gets them to billing. Every meter sends a reading every half hour, so it's a lot of data, around ninety six million readings a day. If it stops, billing doesn't stop straight away, but customers on time-of-use tariffs get estimated bills, which causes complaints.

The readings come in from the meter operator's head-end system as files over SFTP, and a Java ingestion service, that's the one we call the collector, reads the files and puts each reading on Kafka. Kafka runs on six virtual machines in our data centre, not on OpenShift, I always forget why, something about disk performance. Then the validation service, which does run on OpenShift, checks each reading and writes the good ones to Cassandra. Billing reads from Cassandra through a REST API.

Deployment. We build with Jenkins, the images go to our internal Quay registry, and we deploy with Helm charts onto OpenShift. I think the charts are in the platform-charts repo, but don't quote me, it might have moved. Rolling back is helm rollback to the previous release. The Kafka VMs are patched by the infrastructure team, not by us.

Oh, back to the collector for a second. It keeps a checkpoint file per meter operator, so if it restarts it doesn't read the same file twice.

Monitoring is Prometheus and Grafana on OpenShift. The main thing to watch is consumer lag on the validation topic; if it goes over a million messages, an alert goes to the on-call phone through PagerDuty.

What breaks. The collector stops when the head-end sends a file with a corrupt header, it just halts on that file. Move the file to the quarantine folder and the collector carries on. And Cassandra sometimes has a node go down during compaction; usually it comes back on its own, but if it doesn't, restart the Cassandra service on that node.

Backups, so, Cassandra has snapshots every night, kept for a week I believe. Kafka keeps seven days of messages, so in theory we can replay. The failover to the second data centre exists but we haven't documented it, and I'm not sure it works.

On security, the SFTP connection uses key-based authentication, and the keys are in HashiCorp Vault. Meter readings are personal data, so only the billing service account can read Cassandra.

One thing, really important: do not change the number of partitions on the readings topic. Ordering per meter depends on it, and billing will get readings out of order.

The team is the metering platform team, five of us. If you can't reach anyone, go to Sofia Marin, she's the head of metering. Billing problems themselves belong to the billing team.

Things in flight. We're halfway through moving the collector from Java 8 to Java 17. And there's a request from billing to add fifteen-minute readings, which nobody has started.

So for your first week, get Vault access and get added to the PagerDuty schedule. After a month I'd like you to be the first responder on your own.""",
    "facts": [
        {"id": "F01", "quote": "takes the smart meter readings from about two million homes and gets them to billing", "in": ["system_overview"]},
        {"id": "F02", "quote": "Every meter sends a reading every half hour", "in": ["system_overview"]},
        {"id": "F03", "quote": "around ninety six million readings a day", "in": ["system_overview"]},
        {"id": "F04", "quote": "customers on time-of-use tariffs get estimated bills", "in": ["system_overview"]},
        {"id": "F05", "quote": "The readings come in from the meter operator's head-end system as files over SFTP", "in": AR},
        {"id": "F06", "quote": "reads the files and puts each reading on Kafka", "in": AR},
        {"id": "F07", "quote": "Kafka runs on six virtual machines in our data centre", "in": AR},
        {"id": "F08", "quote": "the validation service, which does run on OpenShift, checks each reading", "in": AR},
        {"id": "F09", "quote": "writes the good ones to Cassandra", "in": AR},
        {"id": "F10", "quote": "Billing reads from Cassandra through a REST API", "in": AR},
        {"id": "F11", "quote": "We build with Jenkins", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "the images go to our internal Quay registry", "in": ["deployment_and_rollback", "architecture_reference"]},
        {"id": "F13", "quote": "we deploy with Helm charts onto OpenShift", "in": ["deployment_and_rollback"]},
        {"id": "F14", "quote": "I think the charts are in the platform-charts repo", "in": ["deployment_and_rollback"]},
        {"id": "F15", "quote": "Rolling back is helm rollback to the previous release", "in": ["deployment_and_rollback"]},
        {"id": "F16", "quote": "The Kafka VMs are patched by the infrastructure team", "in": ["ownership_escalation", "architecture_reference"]},
        {"id": "F17", "quote": "It keeps a checkpoint file per meter operator", "in": AR},
        {"id": "F18", "quote": "Monitoring is Prometheus and Grafana on OpenShift", "in": ["monitoring_observability"]},
        {"id": "F19", "quote": "The main thing to watch is consumer lag on the validation topic", "in": ["monitoring_observability"]},
        {"id": "F20", "quote": "an alert goes to the on-call phone through PagerDuty", "in": ["monitoring_observability"]},
        {"id": "F21", "quote": "The collector stops when the head-end sends a file with a corrupt header", "in": ["common_failures"]},
        {"id": "F22", "quote": "Move the file to the quarantine folder and the collector carries on", "in": ["common_failures"]},
        {"id": "F23", "quote": "Cassandra sometimes has a node go down during compaction", "in": ["common_failures"]},
        {"id": "F24", "quote": "restart the Cassandra service on that node", "in": ["common_failures"]},
        {"id": "F25", "quote": "Cassandra has snapshots every night", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "Kafka keeps seven days of messages", "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "The failover to the second data centre exists but we haven't documented it", "in": ["disaster_recovery"]},
        {"id": "F28", "quote": "the SFTP connection uses key-based authentication", "in": ["security_controls"]},
        {"id": "F29", "quote": "the keys are in HashiCorp Vault", "in": ["security_controls"]},
        {"id": "F30", "quote": "only the billing service account can read Cassandra", "in": ["security_controls"]},
        {"id": "F31", "quote": "do not change the number of partitions on the readings topic", "in": ["danger_zones"]},
        {"id": "F32", "quote": "Ordering per meter depends on it", "in": ["danger_zones"]},
        {"id": "F33", "quote": "The team is the metering platform team", "in": ["ownership_escalation"]},
        {"id": "F34", "quote": "go to Sofia Marin, she's the head of metering", "in": ["ownership_escalation"]},
        {"id": "F35", "quote": "Billing problems themselves belong to the billing team", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "moving the collector from Java 8 to Java 17", "in": ["open_responsibilities"]},
        {"id": "F37", "quote": "a request from billing to add fifteen-minute readings, which nobody has started", "in": ["open_responsibilities"]},
        {"id": "F38", "quote": "get Vault access and get added to the PagerDuty schedule", "in": DAY1},
        {"id": "F39", "quote": "be the first responder on your own", "in": MONTH1},
    ],
    "covered": ["system_overview", "architecture_reference", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "ownership_escalation",
                "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": ["cost_optimization", "known_bad_days", "environments"],
})


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bad = 0
    for kt in KTS:
        text = norm(kt["transcript"])
        for f in kt["facts"]:
            if norm(f["quote"]) not in text:
                print("QUOTE NOT IN TRANSCRIPT", kt["id"], f["id"], f["quote"])
                bad += 1
        kt.setdefault("source", S)
        kt["holdout"] = True
        with open(os.path.join(out_dir, kt["id"] + ".json"), "w", encoding="utf-8") as fh:
            json.dump(kt, fh, indent=1, ensure_ascii=False)
    print(len(KTS), "KTs,", sum(len(k["facts"]) for k in KTS), "facts,", bad, "bad quotes")


if __name__ == "__main__":
    main(sys.argv[1])
