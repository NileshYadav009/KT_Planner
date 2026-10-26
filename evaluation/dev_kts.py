"""Round-6 development KTs (written 16 Oct 2026, before any round-6 change).

    python dev_kts.py OUTDIR     # writes one golden JSON per KT, checks quotes

Six styles the existing goldens under-represent: formal monologue, labelled
three-person meeting, speech-to-text noise, read-aloud bullet notes,
incoming-owner Q&A, and a rambling talk that returns to topics. Annotation
follows the tuning goldens' conventions (components: architecture_reference
or system_overview; build and release steps: deployment_and_rollback; ...).

Round 8 (26 Oct, audit §9abd) adds d07-d10 in the four styles of the blind
set b04-b07 (from evaluation/README.md, not from that set).
"""
import json
import os
import re
import sys

S = "written for round 6 (synthetic)"


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(t).lower())).strip()


AR = ["architecture_reference", "system_overview"]
KTS = []

# ---------------------------------------------------------------------------
KTS.append({
    "id": "d01_clearwater_azure_formal",
    "title": "Clearwater: Azure Functions reconciliation, formal monologue",
    "proves": "Formal register, long sentences, cost and calendar discussed.",
    "transcript": """Good afternoon. The purpose of this session is to transfer ownership of Clearwater, the payment reconciliation service used by the treasury department. Clearwater matches incoming bank statements against the ledger every night and flags any payment that cannot be matched. Roughly forty thousand transactions are reconciled each business day, and the treasury analysts rely on the morning exception report to release supplier payments.

The service is built on Azure Functions running in a Premium plan. Bank statement files arrive over SFTP into a Blob Storage container, which triggers the parsing function. Parsed transactions are written to Azure Cosmos DB, and the matching results are stored in Azure SQL Database. Treasury users read the exception report through a Power BI workspace.

There are three environments: development, test, and production. The test environment receives anonymised statement files from two of our banks, so it is safe to replay files there.

Releases are performed with Azure DevOps. A pull request into the main branch runs the unit tests and publishes the function package. Promotion to production requires approval from the treasury systems lead. To roll back, redeploy the previous package from the release history; this completes in approximately five minutes.

Application Insights collects telemetry from every function. An alert fires when the nightly run has not completed by five in the morning, and the alert is routed to the on-call engineer through PagerDuty.

The most frequent incident is a bank changing its file format without notice, which causes the parser to reject the entire file. The remedy is to update the column mapping in the configuration table and re-trigger the parse for that file. A second recurring issue is Cosmos DB throttling during month-end volumes; increase the provisioned throughput temporarily and the backlog clears within an hour.

Cosmos DB has continuous backup with a seven day restore window, and Azure SQL keeps point in time restore for thirty five days. The recovery time objective is four hours. A full restore exercise was last performed in March.

The function apps use managed identities, and all bank credentials are held in Key Vault. Statement files are encrypted at rest.

Under no circumstances should the matching tolerance setting be changed without written approval from treasury, because it directly affects which payments are released.

Month-end and the first two business days of each month are the busiest period; no releases are permitted during that window.

The Premium plan is the largest cost item. We reduced the always-ready instances from three to one outside business hours, which saved roughly a third of the monthly bill.

Ownership sits with the finance platform team. For escalation, contact Helen Oduya, the finance platform manager.

One item remains open: the onboarding of a fourth bank, which has been scoped but not started.

In your first week, obtain contributor access to the production subscription. Within the first month, you should lead one production release under supervision.""",
    "facts": [
        {"id": "F01", "quote": "Clearwater, the payment reconciliation service used by the treasury department", "in": ["system_overview"]},
        {"id": "F02", "quote": "Clearwater matches incoming bank statements against the ledger every night", "in": ["system_overview"]},
        {"id": "F03", "quote": "Roughly forty thousand transactions are reconciled each business day", "in": ["system_overview"]},
        {"id": "F04", "quote": "the treasury analysts rely on the morning exception report to release supplier payments", "in": ["system_overview"]},
        {"id": "F05", "quote": "The service is built on Azure Functions running in a Premium plan", "in": AR},
        {"id": "F06", "quote": "Bank statement files arrive over SFTP into a Blob Storage container", "in": AR},
        {"id": "F07", "quote": "Parsed transactions are written to Azure Cosmos DB", "in": AR},
        {"id": "F08", "quote": "the matching results are stored in Azure SQL Database", "in": AR},
        {"id": "F09", "quote": "Treasury users read the exception report through a Power BI workspace", "in": AR},
        {"id": "F10", "quote": "There are three environments: development, test, and production", "in": ["environments"]},
        {"id": "F11", "quote": "The test environment receives anonymised statement files from two of our banks", "in": ["environments"]},
        {"id": "F12", "quote": "Releases are performed with Azure DevOps", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "A pull request into the main branch runs the unit tests and publishes the function package", "in": ["deployment_and_rollback"]},
        {"id": "F14", "quote": "Promotion to production requires approval from the treasury systems lead", "in": ["deployment_and_rollback"]},
        {"id": "F15", "quote": "redeploy the previous package from the release history", "in": ["deployment_and_rollback"]},
        {"id": "F16", "quote": "Application Insights collects telemetry from every function", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "An alert fires when the nightly run has not completed by five in the morning", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "a bank changing its file format without notice, which causes the parser to reject the entire file", "in": ["common_failures"]},
        {"id": "F19", "quote": "update the column mapping in the configuration table and re-trigger the parse for that file", "in": ["common_failures"]},
        {"id": "F20", "quote": "Cosmos DB throttling during month-end volumes", "in": ["common_failures", "known_bad_days"]},
        {"id": "F21", "quote": "increase the provisioned throughput temporarily and the backlog clears within an hour", "in": ["common_failures"]},
        {"id": "F22", "quote": "Cosmos DB has continuous backup with a seven day restore window", "in": ["disaster_recovery"]},
        {"id": "F23", "quote": "Azure SQL keeps point in time restore for thirty five days", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "The recovery time objective is four hours", "alt": ["four hours"], "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "A full restore exercise was last performed in March", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "The function apps use managed identities", "in": ["security_controls"]},
        {"id": "F27", "quote": "all bank credentials are held in Key Vault", "in": ["security_controls"]},
        {"id": "F28", "quote": "Statement files are encrypted at rest", "in": ["security_controls"]},
        {"id": "F29", "quote": "should the matching tolerance setting be changed without written approval from treasury", "in": ["danger_zones"]},
        {"id": "F30", "quote": "Month-end and the first two business days of each month are the busiest period", "in": ["known_bad_days"]},
        {"id": "F31", "quote": "no releases are permitted during that window", "in": ["known_bad_days", "deployment_and_rollback"]},
        {"id": "F32", "quote": "The Premium plan is the largest cost item", "in": ["cost_optimization"]},
        {"id": "F33", "quote": "We reduced the always-ready instances from three to one outside business hours", "in": ["cost_optimization"]},
        {"id": "F34", "quote": "Ownership sits with the finance platform team", "in": ["ownership_escalation"]},
        {"id": "F35", "quote": "contact Helen Oduya, the finance platform manager", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "the onboarding of a fourth bank, which has been scoped but not started", "in": ["open_responsibilities"]},
        {"id": "F37", "quote": "obtain contributor access to the production subscription", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F38", "quote": "you should lead one production release under supervision", "in": ["first_30_day_ownership", "handover_completion"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "known_bad_days", "cost_optimization",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": [],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "d02_sparrow_gke_meeting",
    "title": "Sparrow: GKE recommendations API, labelled three-person meeting",
    "proves": "Speaker labels, interruptions, short turns, facts split across speakers.",
    "transcript": """Priya: Okay, I think everyone's here. Marcus, this is the handover for Sparrow, since you're picking it up from Monday.
Marcus: Great, thanks.
Priya: So Sparrow is the product recommendations API for the online store. Every product page calls it, so if it's slow the whole page is slow.
Dan: And it drives about twelve percent of revenue through the "you might also like" carousel, so it matters.
Priya: Right. It runs on GKE Autopilot. There's a Python FastAPI service that serves the recommendations, and a nightly training job on Vertex AI that rebuilds the model.
Dan: The features come from BigQuery, and the serving side keeps the latest recommendations in Memorystore Redis.
Marcus: How do releases work?
Priya: Cloud Build builds the image on every merge, and Argo CD syncs it to the cluster. For a rollback you just revert the commit in the config repo and Argo puts the old version back.
Dan: Model releases are separate. A new model only goes live if its offline click-through score beats the current one.
Marcus: What environments are there?
Priya: Dev and prod only. There's no staging, which honestly is a gap.
Marcus: Monitoring?
Dan: Cloud Monitoring dashboards for latency and error rate, and we alert in Slack when p95 latency goes over two hundred milliseconds.
Priya: The thing that goes wrong most is the training job failing because a BigQuery table schema changed upstream. When that happens the API keeps serving yesterday's recommendations, so it's not urgent, but you need to fix the query and rerun the job.
Dan: The other one is Redis evicting keys when memory fills up, and then latency spikes. Bump the Memorystore tier and it settles.
Marcus: Backups?
Priya: The models are versioned in a Cloud Storage bucket, so we can always redeploy an older model. BigQuery has time travel for seven days. We've never tested a full region failure.
Dan: Security wise, the service account only has read access to BigQuery, and the API sits behind Identity-Aware Proxy for the admin endpoints.
Priya: Please don't delete the old model versions from the bucket, the rollback depends on them.
Marcus: Who do I go to when things break?
Priya: The personalisation team owns Sparrow. Escalate to Dan first, then to Grace Lim, the head of data science.
Dan: There's also an unfinished A/B test framework migration. We moved half the experiments to the new framework and the rest are still on the old one.
Priya: For your first day, get added to the sparrow-oncall Slack group and the GCP project. By the end of the month you should be doing the model releases on your own.""",
    "facts": [
        {"id": "F01", "quote": "Sparrow is the product recommendations API for the online store", "in": ["system_overview"]},
        {"id": "F02", "quote": "Every product page calls it, so if it's slow the whole page is slow", "in": ["system_overview"]},
        {"id": "F03", "quote": "it drives about twelve percent of revenue", "in": ["system_overview"]},
        {"id": "F04", "quote": "It runs on GKE Autopilot", "in": AR},
        {"id": "F05", "quote": "a Python FastAPI service that serves the recommendations", "in": AR},
        {"id": "F06", "quote": "a nightly training job on Vertex AI that rebuilds the model", "in": AR},
        {"id": "F07", "quote": "The features come from BigQuery", "in": AR},
        {"id": "F08", "quote": "the serving side keeps the latest recommendations in Memorystore Redis", "in": AR},
        {"id": "F09", "quote": "Cloud Build builds the image on every merge", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "Argo CD syncs it to the cluster", "alt": ["ArgoCD syncs it to the cluster"], "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "revert the commit in the config repo", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "A new model only goes live if its offline click-through score beats the current one", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "Dev and prod only", "in": ["environments"]},
        {"id": "F14", "quote": "There's no staging", "in": ["environments"]},
        {"id": "F15", "quote": "Cloud Monitoring dashboards for latency and error rate", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "we alert in Slack when p95 latency goes over two hundred milliseconds", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "the training job failing because a BigQuery table schema changed upstream", "in": ["common_failures"]},
        {"id": "F18", "quote": "the API keeps serving yesterday's recommendations", "in": ["common_failures"]},
        {"id": "F19", "quote": "you need to fix the query and rerun the job", "in": ["common_failures"]},
        {"id": "F20", "quote": "Redis evicting keys when memory fills up, and then latency spikes", "in": ["common_failures"]},
        {"id": "F21", "quote": "Bump the Memorystore tier and it settles", "in": ["common_failures"]},
        {"id": "F22", "quote": "The models are versioned in a Cloud Storage bucket", "in": ["disaster_recovery", "deployment_and_rollback"]},
        {"id": "F23", "quote": "BigQuery has time travel for seven days", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "We've never tested a full region failure", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "the service account only has read access to BigQuery", "in": ["security_controls"]},
        {"id": "F26", "quote": "the API sits behind Identity-Aware Proxy for the admin endpoints", "in": ["security_controls"]},
        {"id": "F27", "quote": "don't delete the old model versions from the bucket", "in": ["danger_zones"]},
        {"id": "F28", "quote": "The personalisation team owns Sparrow", "in": ["ownership_escalation"], "terms": ["personalisation team", "sparrow"]},
        {"id": "F29", "quote": "Escalate to Dan first, then to Grace Lim, the head of data science", "in": ["ownership_escalation"]},
        {"id": "F30", "quote": "an unfinished A/B test framework migration", "in": ["open_responsibilities"]},
        {"id": "F31", "quote": "We moved half the experiments to the new framework and the rest are still on the old one", "in": ["open_responsibilities"]},
        {"id": "F32", "quote": "get added to the sparrow-oncall Slack group and the GCP project", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F33", "quote": "By the end of the month you should be doing the model releases on your own", "in": ["first_30_day_ownership"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "ownership_escalation",
                "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": ["cost_optimization", "known_bad_days"],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "d03_tollgate_lambda_noisy",
    "title": "Tollgate: AWS Lambda toll events, speech-to-text noise",
    "proves": "Fillers, run-on sentences, self-corrections, lowercase spoken style.",
    "transcript": """um okay so yeah this is the KT for tollgate. so tollgate is basically the thing that takes the toll plaza events, like every car that goes through a gantry, and um turns them into charges on the customer's account. we're talking like two million events a day, more on bank holidays obviously.

so architecture wise, the gantries send events to API Gateway, and then a Lambda function validates them and drops them onto a Kinesis stream. uh another set of Lambdas reads from Kinesis and writes the charges into DynamoDB. and there's a nightly batch on AWS Batch that builds the invoices and puts the PDFs in S3.

deployments, so we use the Serverless Framework, and um the GitHub Actions pipeline runs serverless deploy to staging when you merge, and then prod is a manual workflow run. if something's broken after a deploy you can just, you know, run the deploy again with the previous git tag, that's our rollback basically.

environments we've got dev, staging and prod. staging uses recorded gantry traffic from last year so you can replay a busy day there.

for monitoring it's all CloudWatch. there's a dashboard called tollgate overview, and um the main alarm is the Kinesis iterator age, if that goes above five minutes it pages whoever's on call through Opsgenie.

okay so what breaks. the big one is Lambda hitting the concurrency limit when there's a surge, like after a gantry comes back online and dumps its buffer. then events get throttled. what you do is raise the reserved concurrency for the ingest function, it's a setting in the serverless yaml. the other thing is DynamoDB hot partitions on the busiest plazas, you'll see throttled writes on just one or two plaza ids. we added write sharding for the worst ones but um not all of them.

backups, DynamoDB has point in time recovery on, and the S3 invoice bucket has versioning. honestly we haven't written down an RPO.

uh security, the gantries authenticate with mutual TLS certificates, and the certs are rotated every year by the roadside team. don't ever, like ever, disable the duplicate check in the validator, we'd double charge people and that's a regulatory problem.

owners, it's the road charging team, i'm the tech lead until friday and then it's you. if it's really bad call Sam Achebe, he's the head of engineering.

oh and one thing, there's a half finished migration from the old Node 14 runtime to Node 20, three functions are still on fourteen.

first thing, get yourself access to the AWS prod account and the Opsgenie rota, and um maybe spend the first couple of weeks just watching the dashboard during the morning rush.""",
    "facts": [
        {"id": "F01", "quote": "takes the toll plaza events", "in": ["system_overview"]},
        {"id": "F02", "quote": "turns them into charges on the customer's account", "in": ["system_overview"]},
        {"id": "F03", "quote": "two million events a day", "in": ["system_overview"]},
        {"id": "F04", "quote": "the gantries send events to API Gateway", "in": AR},
        {"id": "F05", "quote": "a Lambda function validates them and drops them onto a Kinesis stream", "in": AR},
        {"id": "F06", "quote": "another set of Lambdas reads from Kinesis and writes the charges into DynamoDB", "in": AR},
        {"id": "F07", "quote": "a nightly batch on AWS Batch that builds the invoices and puts the PDFs in S3", "in": AR},
        {"id": "F08", "quote": "we use the Serverless Framework", "in": ["deployment_and_rollback", "architecture_reference"]},
        {"id": "F09", "quote": "runs serverless deploy to staging when you merge", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "prod is a manual workflow run", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "run the deploy again with the previous git tag", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "we've got dev, staging and prod", "in": ["environments"]},
        {"id": "F13", "quote": "staging uses recorded gantry traffic from last year", "in": ["environments"]},
        {"id": "F14", "quote": "for monitoring it's all CloudWatch", "alt": ["it's all CloudWatch"], "in": ["monitoring_observability"]},
        {"id": "F15", "quote": "a dashboard called tollgate overview", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "the main alarm is the Kinesis iterator age", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "it pages whoever's on call through Opsgenie", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "Lambda hitting the concurrency limit when there's a surge", "in": ["common_failures"]},
        {"id": "F19", "quote": "events get throttled", "in": ["common_failures"]},
        {"id": "F20", "quote": "raise the reserved concurrency for the ingest function", "in": ["common_failures"]},
        {"id": "F21", "quote": "DynamoDB hot partitions on the busiest plazas", "in": ["common_failures"]},
        {"id": "F22", "quote": "we added write sharding for the worst ones", "in": ["common_failures"]},
        {"id": "F23", "quote": "DynamoDB has point in time recovery on", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "the S3 invoice bucket has versioning", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "we haven't written down an RPO", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "the gantries authenticate with mutual TLS certificates", "in": ["security_controls"]},
        {"id": "F27", "quote": "the certs are rotated every year by the roadside team", "in": ["security_controls"]},
        {"id": "F28", "quote": "disable the duplicate check in the validator", "in": ["danger_zones"]},
        {"id": "F29", "quote": "it's the road charging team", "in": ["ownership_escalation"]},
        {"id": "F30", "quote": "call Sam Achebe, he's the head of engineering", "in": ["ownership_escalation"]},
        {"id": "F31", "quote": "a half finished migration from the old Node 14 runtime to Node 20", "in": ["open_responsibilities"]},
        {"id": "F32", "quote": "get yourself access to the AWS prod account and the Opsgenie rota", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F33", "quote": "spend the first couple of weeks just watching the dashboard during the morning rush", "in": ["first_30_day_ownership", "day1_survival_checklist"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "ownership_escalation",
                "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": ["cost_optimization"],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "d04_vault_openshift_notes",
    "title": "Vaultline: on-prem OpenShift core banking adapter, read-aloud notes",
    "proves": "Terse note fragments with headings spoken aloud.",
    "transcript": """Handover notes for Vaultline. Reading these out as they are.

Overview. Vaultline is the adapter between the mobile banking app and the core banking mainframe. It translates REST calls into mainframe transactions. Used by about one point two million retail customers. If it is down, customers cannot see balances or make transfers.

Architecture. Java Spring Boot services on Red Hat OpenShift in our two data centres. Kafka for the event feed to the fraud system. Oracle database for the session and audit tables. IBM MQ to the mainframe.

Environments. Four of them: dev, SIT, UAT and production. UAT is connected to a mainframe test region, not the real one.

Release process. Jenkins pipeline builds and runs the tests. Deployment to production is by change request only, approved at the Tuesday change board. Releases go out on Thursday nights. Rollback is to scale the previous deployment back up in OpenShift and scale the new one down.

Monitoring. Dynatrace for traces and service health. Splunk for logs. Alerts go to the operations bridge, who call the on-call developer.

Known problems. MQ channel stops after a mainframe restart. Symptom is every transfer timing out. Fix: restart the channel from the MQ console. Second one, Oracle sessions table grows and slows logins. Run the purge job, it is safe in business hours.

Backups and DR. Oracle backed up nightly with RMAN. Active-passive across the two data centres. Failover drill twice a year. RTO thirty minutes, RPO five minutes.

Security. All services behind the API gateway with OAuth tokens. Penetration test every year by an external firm. Secrets in CyberArk.

Do not touch. Never change the MQ queue manager names. Never run the Oracle purge with the full flag.

Calendar. Salary days, the twenty fifth to the twenty eighth, are the heaviest. No changes in the last week of December.

Ownership. Digital channels team owns Vaultline. Mainframe team owns the MQ side. Escalation: on-call developer, then Farid Haddad, then the head of digital.

Open items. TLS upgrade on the MQ channels is waiting on the mainframe team. Kafka upgrade not started.

First steps. Day one, request OpenShift and Splunk access. First month, attend two change boards and shadow one Thursday release.""",
    "facts": [
        {"id": "F01", "quote": "Vaultline is the adapter between the mobile banking app and the core banking mainframe", "in": ["system_overview"]},
        {"id": "F02", "quote": "It translates REST calls into mainframe transactions", "in": ["system_overview", "architecture_reference"]},
        {"id": "F03", "quote": "Used by about one point two million retail customers", "in": ["system_overview"]},
        {"id": "F04", "quote": "customers cannot see balances or make transfers", "in": ["system_overview"]},
        {"id": "F05", "quote": "Java Spring Boot services on Red Hat OpenShift in our two data centres", "in": AR},
        {"id": "F06", "quote": "Kafka for the event feed to the fraud system", "in": AR},
        {"id": "F07", "quote": "Oracle database for the session and audit tables", "in": AR},
        {"id": "F08", "quote": "IBM MQ to the mainframe", "in": AR},
        {"id": "F09", "quote": "Four of them: dev, SIT, UAT and production", "in": ["environments"]},
        {"id": "F10", "quote": "UAT is connected to a mainframe test region, not the real one", "in": ["environments"]},
        {"id": "F11", "quote": "Jenkins pipeline builds and runs the tests", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "Deployment to production is by change request only, approved at the Tuesday change board", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "Releases go out on Thursday nights", "in": ["deployment_and_rollback", "known_bad_days"]},
        {"id": "F14", "quote": "scale the previous deployment back up in OpenShift and scale the new one down", "in": ["deployment_and_rollback"]},
        {"id": "F15", "quote": "Dynatrace for traces and service health", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "Splunk for logs", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "Alerts go to the operations bridge, who call the on-call developer", "in": ["monitoring_observability", "ownership_escalation"]},
        {"id": "F18", "quote": "MQ channel stops after a mainframe restart", "in": ["common_failures"]},
        {"id": "F19", "quote": "every transfer timing out", "in": ["common_failures"]},
        {"id": "F20", "quote": "restart the channel from the MQ console", "in": ["common_failures"]},
        {"id": "F21", "quote": "Oracle sessions table grows and slows logins", "in": ["common_failures"]},
        {"id": "F22", "quote": "Run the purge job, it is safe in business hours", "in": ["common_failures"]},
        {"id": "F23", "quote": "Oracle backed up nightly with RMAN", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "Active-passive across the two data centres", "in": ["disaster_recovery", "architecture_reference"]},
        {"id": "F25", "quote": "Failover drill twice a year", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "RTO thirty minutes", "alt": ["thirty minutes"], "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "RPO five minutes", "alt": ["five minutes"], "in": ["disaster_recovery"]},
        {"id": "F28", "quote": "All services behind the API gateway with OAuth tokens", "in": ["security_controls"]},
        {"id": "F29", "quote": "Penetration test every year by an external firm", "in": ["security_controls"]},
        {"id": "F30", "quote": "Secrets in CyberArk", "in": ["security_controls"]},
        {"id": "F31", "quote": "Never change the MQ queue manager names", "in": ["danger_zones"]},
        {"id": "F32", "quote": "Never run the Oracle purge with the full flag", "in": ["danger_zones"]},
        {"id": "F33", "quote": "Salary days, the twenty fifth to the twenty eighth, are the heaviest", "in": ["known_bad_days"]},
        {"id": "F34", "quote": "No changes in the last week of December", "in": ["known_bad_days", "deployment_and_rollback"]},
        {"id": "F35", "quote": "Digital channels team owns Vaultline", "in": ["ownership_escalation"], "terms": ["digital channels team", "vaultline"]},
        {"id": "F36", "quote": "Mainframe team owns the MQ side", "in": ["ownership_escalation"], "terms": ["mainframe team", "mq"]},
        {"id": "F37", "quote": "on-call developer, then Farid Haddad, then the head of digital", "in": ["ownership_escalation"]},
        {"id": "F38", "quote": "TLS upgrade on the MQ channels is waiting on the mainframe team", "in": ["open_responsibilities"]},
        {"id": "F39", "quote": "Kafka upgrade not started", "in": ["open_responsibilities"]},
        {"id": "F40", "quote": "request OpenShift and Splunk access", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F41", "quote": "attend two change boards and shadow one Thursday release", "in": ["first_30_day_ownership"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "known_bad_days",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": ["cost_optimization"],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "d05_fernhill_hybrid_qa",
    "title": "Fernhill: hybrid clinic booking, incoming owner asks the questions",
    "proves": "The incoming owner drives with questions; answers are short and rely on the question.",
    "transcript": """Nadia: Before you go, can I run through my list of questions about Fernhill?
Owen: Sure, go ahead.
Nadia: What is Fernhill exactly?
Owen: It's the appointment booking system for our forty clinics. Patients book online or through the call centre, and the clinics see their day's list on a tablet.
Nadia: How many bookings is that?
Owen: Around eight thousand a day, and Monday mornings are by far the busiest.
Nadia: Where does it run?
Owen: It's hybrid. The web front end and the booking API run on Azure App Service. The patient records stay on premises in a SQL Server cluster, and the API reaches them over a site-to-site VPN.
Nadia: And messaging?
Owen: Reminder texts go out through Twilio, triggered by an Azure Service Bus queue.
Nadia: How do I deploy?
Owen: Merge to main and GitHub Actions deploys to a staging slot. Check the smoke tests, then swap the slot into production. If it goes wrong, swap back. It takes a minute.
Nadia: Which environments do we have?
Owen: Dev, staging and production. Staging points at a test copy of the clinic database with fake patients.
Nadia: How will I know something is wrong?
Owen: Azure Monitor alerts on failed bookings, and there's a Grafana board on the on-prem side for the SQL cluster. Alerts land in the Teams channel and page the on-call phone.
Nadia: What breaks most often?
Owen: The VPN tunnel drops, and then every booking fails with a database timeout. Reset the tunnel from the Azure portal; if it doesn't come back, call the network team.
Nadia: Anything else?
Owen: Twilio sometimes rejects messages when we hit the rate limit after a mass reschedule. Spread the sends over an hour.
Nadia: What about backups?
Owen: SQL Server does full backups nightly and log backups every fifteen minutes, copied to the second site. We restored a copy in June to test it.
Nadia: Security?
Owen: Patient data never leaves the on-prem cluster except through the API. Staff sign in with Entra ID and multi-factor authentication. We had a data protection audit last spring.
Nadia: What should I never do?
Owen: Never run a schema change on the clinic database during clinic hours. And don't reset the VPN shared key, it has to be changed on both ends at once.
Nadia: Costs?
Owen: The App Service plan is oversized. We could probably drop one tier, but nobody has tested it.
Nadia: Bad days to deploy?
Owen: Avoid Monday mornings and the first week of January, when everyone rebooks.
Nadia: Who owns what?
Owen: The patient access team owns Fernhill. Infrastructure owns the VPN and the SQL cluster. Escalate to Laura Pike, the clinical systems manager.
Nadia: Anything unfinished?
Owen: The move from SMS reminders to app notifications is half done. Only six clinics have it.
Nadia: And my first week?
Owen: Get access to the Azure subscription and the on-prem jump host. Then sit with the call centre for an hour to see how they use it.""",
    "facts": [
        {"id": "F01", "quote": "the appointment booking system for our forty clinics", "in": ["system_overview"]},
        {"id": "F02", "quote": "Patients book online or through the call centre", "in": ["system_overview"]},
        {"id": "F03", "quote": "the clinics see their day's list on a tablet", "in": ["system_overview"]},
        {"id": "F04", "quote": "Around eight thousand a day", "in": ["system_overview"]},
        {"id": "F05", "quote": "Monday mornings are by far the busiest", "in": ["known_bad_days", "system_overview"]},
        {"id": "F06", "quote": "The web front end and the booking API run on Azure App Service", "alt": ["The web frontend and the booking API run on Azure App Service"], "in": AR},
        {"id": "F07", "quote": "The patient records stay on premises in a SQL Server cluster", "in": AR},
        {"id": "F08", "quote": "the API reaches them over a site-to-site VPN", "in": AR},
        {"id": "F09", "quote": "Reminder texts go out through Twilio, triggered by an Azure Service Bus queue", "in": AR},
        {"id": "F10", "quote": "Merge to main and GitHub Actions deploys to a staging slot", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "Check the smoke tests, then swap the slot into production", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "If it goes wrong, swap back", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "Dev, staging and production", "in": ["environments"]},
        {"id": "F14", "quote": "Staging points at a test copy of the clinic database with fake patients", "in": ["environments"]},
        {"id": "F15", "quote": "Azure Monitor alerts on failed bookings", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "a Grafana board on the on-prem side for the SQL cluster", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "Alerts land in the Teams channel and page the on-call phone", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "The VPN tunnel drops, and then every booking fails with a database timeout", "in": ["common_failures"]},
        {"id": "F19", "quote": "Reset the tunnel from the Azure portal", "in": ["common_failures"]},
        {"id": "F20", "quote": "if it doesn't come back, call the network team", "in": ["common_failures", "ownership_escalation"]},
        {"id": "F21", "quote": "Twilio sometimes rejects messages when we hit the rate limit after a mass reschedule", "in": ["common_failures"]},
        {"id": "F22", "quote": "Spread the sends over an hour", "in": ["common_failures"]},
        {"id": "F23", "quote": "SQL Server does full backups nightly and log backups every fifteen minutes", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "copied to the second site", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "We restored a copy in June to test it", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "Patient data never leaves the on-prem cluster except through the API", "in": ["security_controls"]},
        {"id": "F27", "quote": "Staff sign in with Entra ID and multi-factor authentication", "in": ["security_controls"]},
        {"id": "F28", "quote": "We had a data protection audit last spring", "in": ["security_controls"]},
        {"id": "F29", "quote": "Never run a schema change on the clinic database during clinic hours", "in": ["danger_zones", "known_bad_days"]},
        {"id": "F30", "quote": "don't reset the VPN shared key", "in": ["danger_zones"]},
        {"id": "F31", "quote": "The App Service plan is oversized", "in": ["cost_optimization"]},
        {"id": "F32", "quote": "We could probably drop one tier, but nobody has tested it", "in": ["cost_optimization", "open_responsibilities"]},
        {"id": "F33", "quote": "Avoid Monday mornings and the first week of January", "in": ["known_bad_days", "deployment_and_rollback"]},
        {"id": "F34", "quote": "The patient access team owns Fernhill", "in": ["ownership_escalation"], "terms": ["patient access team", "fernhill"]},
        {"id": "F35", "quote": "Infrastructure owns the VPN and the SQL cluster", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "Escalate to Laura Pike, the clinical systems manager", "in": ["ownership_escalation"]},
        {"id": "F37", "quote": "The move from SMS reminders to app notifications is half done", "in": ["open_responsibilities"]},
        {"id": "F38", "quote": "Only six clinics have it", "in": ["open_responsibilities"]},
        {"id": "F39", "quote": "Get access to the Azure subscription and the on-prem jump host", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F40", "quote": "sit with the call centre for an hour to see how they use it", "in": ["first_30_day_ownership", "day1_survival_checklist"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "known_bad_days", "cost_optimization",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": [],
})

# ---------------------------------------------------------------------------
KTS.append({
    "id": "d06_brightline_airflow_ramble",
    "title": "Brightline: Airflow and Snowflake data pipeline, rambling talk with topic returns",
    "proves": "Topics revisited out of order; failure stories told inside other topics.",
    "transcript": """Alright, so Brightline. Brightline is our data pipeline that loads sales and inventory data from the stores into Snowflake, and the finance and merchandising teams build all their reports on top of it. If the morning load is late, the daily sales meeting has no numbers.

So the way it works, Airflow on Amazon MWAA runs the DAGs. The stores push CSV extracts to an S3 landing bucket overnight, then dbt models transform the raw tables into the reporting schema in Snowflake. Tableau dashboards read from the reporting schema.

Monitoring, we have Airflow's own task alerts going to a Slack channel, and there's a Monte Carlo data quality check on the main tables. Actually that reminds me, the most annoying failure: a store sends an empty file, the load succeeds, and then the sales numbers for that store show as zero. Monte Carlo catches it as a volume anomaly. You re-request the file from the store's IT contact and rerun the DAG for that date.

Deployments are pretty simple. The DAGs and dbt models live in one GitHub repo, a merge to main runs dbt tests in CI and then syncs the DAG folder to the MWAA bucket. If a model is wrong, revert the merge and rerun the affected models.

We have a dev Snowflake database and production, and dev is a zero-copy clone of production refreshed weekly.

Going back to failures, the other common one is the Snowflake warehouse queueing at seven in the morning when everyone opens Tableau at once. Scale the reporting warehouse up to large for an hour. It's in the runbook.

Costs, Snowflake is most of it. The warehouses auto-suspend after five minutes, and we moved the heavy dbt runs to a separate warehouse so we can see what they cost.

For recovery, Snowflake time travel is set to thirty days on the reporting schema, and the raw files stay in S3 for a year, so we can always rebuild. We've never had to rebuild everything though.

Security, access to Snowflake is through Okta single sign-on, and the store-level margin columns are masked for everyone outside finance.

Please never drop or replace the raw schema tables, the time travel there is only one day. And do not change the DAG start dates, Airflow will try to backfill years of data.

Year-end stock take in the first week of January is the worst week, everything gets reloaded. Black Friday too.

Ownership is the data engineering team, and the dbt models for finance are co-owned with the finance analytics team. For escalation go to Mei Tanaka, the data platform lead.

Unfinished: the migration of the last ten legacy SQL scripts into dbt. They still run from a cron box under someone's desk, literally.

On day one, get Snowflake and MWAA access and join the alerts channel. For the first month, own the morning load check every day.""",
    "facts": [
        {"id": "F01", "quote": "loads sales and inventory data from the stores into Snowflake", "in": ["system_overview"]},
        {"id": "F02", "quote": "the finance and merchandising teams build all their reports on top of it", "in": ["system_overview"]},
        {"id": "F03", "quote": "If the morning load is late, the daily sales meeting has no numbers", "in": ["system_overview"]},
        {"id": "F04", "quote": "Airflow on Amazon MWAA runs the DAGs", "in": AR},
        {"id": "F05", "quote": "The stores push CSV extracts to an S3 landing bucket overnight", "in": AR},
        {"id": "F06", "quote": "dbt models transform the raw tables into the reporting schema in Snowflake", "in": AR},
        {"id": "F07", "quote": "Tableau dashboards read from the reporting schema", "in": AR},
        {"id": "F08", "quote": "Airflow's own task alerts going to a Slack channel", "in": ["monitoring_observability"]},
        {"id": "F09", "quote": "a Monte Carlo data quality check on the main tables", "in": ["monitoring_observability"]},
        {"id": "F10", "quote": "a store sends an empty file, the load succeeds", "in": ["common_failures"]},
        {"id": "F11", "quote": "the sales numbers for that store show as zero", "in": ["common_failures"]},
        {"id": "F12", "quote": "Monte Carlo catches it as a volume anomaly", "in": ["common_failures", "monitoring_observability"]},
        {"id": "F13", "quote": "re-request the file from the store's IT contact and rerun the DAG for that date", "in": ["common_failures"]},
        {"id": "F14", "quote": "The DAGs and dbt models live in one GitHub repo", "in": ["deployment_and_rollback", "architecture_reference"]},
        {"id": "F15", "quote": "a merge to main runs dbt tests in CI and then syncs the DAG folder to the MWAA bucket", "in": ["deployment_and_rollback"]},
        {"id": "F16", "quote": "revert the merge and rerun the affected models", "in": ["deployment_and_rollback"]},
        {"id": "F17", "quote": "We have a dev Snowflake database and production", "in": ["environments"]},
        {"id": "F18", "quote": "dev is a zero-copy clone of production refreshed weekly", "in": ["environments"]},
        {"id": "F19", "quote": "the Snowflake warehouse queueing at seven in the morning", "in": ["common_failures"]},
        {"id": "F20", "quote": "Scale the reporting warehouse up to large for an hour", "in": ["common_failures"]},
        {"id": "F21", "quote": "Snowflake is most of it", "in": ["cost_optimization"]},
        {"id": "F22", "quote": "The warehouses auto-suspend after five minutes", "in": ["cost_optimization"]},
        {"id": "F23", "quote": "we moved the heavy dbt runs to a separate warehouse so we can see what they cost", "in": ["cost_optimization"]},
        {"id": "F24", "quote": "Snowflake time travel is set to thirty days on the reporting schema", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "the raw files stay in S3 for a year", "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "We've never had to rebuild everything", "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "access to Snowflake is through Okta single sign-on", "in": ["security_controls"]},
        {"id": "F28", "quote": "the store-level margin columns are masked for everyone outside finance", "in": ["security_controls"]},
        {"id": "F29", "quote": "never drop or replace the raw schema tables", "in": ["danger_zones"]},
        {"id": "F30", "quote": "do not change the DAG start dates", "in": ["danger_zones"]},
        {"id": "F31", "quote": "Year-end stock take in the first week of January is the worst week", "in": ["known_bad_days"]},
        {"id": "F32", "quote": "Black Friday too", "in": ["known_bad_days"]},
        {"id": "F33", "quote": "Ownership is the data engineering team", "in": ["ownership_escalation"]},
        {"id": "F34", "quote": "the dbt models for finance are co-owned with the finance analytics team", "in": ["ownership_escalation"]},
        {"id": "F35", "quote": "go to Mei Tanaka, the data platform lead", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "the migration of the last ten legacy SQL scripts into dbt", "in": ["open_responsibilities"]},
        {"id": "F37", "quote": "They still run from a cron box under someone's desk", "in": ["open_responsibilities"]},
        {"id": "F38", "quote": "get Snowflake and MWAA access and join the alerts channel", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F39", "quote": "own the morning load check every day", "in": ["first_30_day_ownership"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "known_bad_days", "cost_optimization",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": [],
})

# ---------------------------------------------------------------------------
# Round-8 additions (26 Oct 2026, audit §9abd): the four styles of the blind
# set b04-b07, taken from evaluation/README.md only, so that its discussed
# sections reported Missing (X67) can be looked for on KTs that may be read.
ALL15 = ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
         "common_failures", "disaster_recovery", "security_controls", "danger_zones", "known_bad_days", "cost_optimization",
         "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"]
DAY1 = ["day1_survival_checklist", "first_30_day_ownership"]
MONTH1 = ["first_30_day_ownership", "handover_completion"]

KTS.append({
    "id": "d07_millbrook_onprem_qa",
    "title": "Millbrook: on-premises student records, questions and answers without labels",
    "proves": "Each topic opened by the receiver's question, answered in the same paragraph; no speaker labels.",
    "transcript": """So what is Millbrook exactly? Millbrook is the student records system for the university. Registry staff use it to enrol students, record marks and produce transcripts. About thirty thousand students are in it at any time.

What does it run on? It's all on premises. The application is Java running on Oracle WebLogic, on four Red Hat virtual machines. The database is Oracle 19c on a two-node RAC cluster. Students see their marks through the portal, which calls Millbrook's REST API. Nightly, a batch job exports enrolments to the finance system as a CSV file.

How do changes get deployed? We use Ansible playbooks from the GitLab repository. A change goes to the test server first, and the registry team signs it off before it goes live. Deployments happen on Thursday evenings. If something goes wrong, we run the rollback playbook, which puts back the previous WAR file.

Are there other environments? There's test and production, plus a training copy that the registry uses for new staff.

How do you know if it's broken? Nagios checks the WebLogic servers and the database listener every five minutes. If the portal can't reach the API, Nagios emails the infrastructure team and sends a text to the on-call person.

What usually goes wrong? The WebLogic managed servers run out of heap during results week, and pages time out. We restart the managed servers one at a time so nobody notices. The other one is the finance export failing when a student record has a comma in the address; we fix the record and rerun the export by hand.

What about backups? RMAN takes a full backup every Sunday and incrementals every night, and they're copied to the second data centre. We can restore within six hours. The last restore test was in the summer.

Is there anything about security I should know? Staff sign in with their university account through Shibboleth. Only registry managers can change a final mark, and every change is logged. Student data must never leave the UK.

Anything I should never do? Never run the end-of-year rollover job twice. It advances every student a year, and undoing it means a database restore.

When are the busy times? Results week in June is the worst, then enrolment in September. Avoid any change in those weeks.

Who owns it? The student systems team in IT owns Millbrook. The registry owns the data. If you need to escalate, go to Peter Hall, the head of student systems.

What's still in progress? We're replacing the finance CSV export with an API, and it's about half done. Also the Oracle licence renewal is due in December.

What should I do first? In the first week, get an account on the WebLogic console and read-only access to the database. By the end of the month, run a Thursday deployment yourself.""",
    "facts": [
        {"id": "F01", "quote": "Millbrook is the student records system for the university", "in": ["system_overview"]},
        {"id": "F02", "quote": "Registry staff use it to enrol students, record marks and produce transcripts", "in": ["system_overview"]},
        {"id": "F03", "quote": "About thirty thousand students are in it at any time", "in": ["system_overview"]},
        {"id": "F04", "quote": "It's all on premises", "in": AR},
        {"id": "F05", "quote": "The application is Java running on Oracle WebLogic, on four Red Hat virtual machines", "in": AR},
        {"id": "F06", "quote": "The database is Oracle 19c on a two-node RAC cluster", "in": AR},
        {"id": "F07", "quote": "Students see their marks through the portal, which calls Millbrook's REST API", "in": AR},
        {"id": "F08", "quote": "a batch job exports enrolments to the finance system as a CSV file", "in": AR},
        {"id": "F09", "quote": "We use Ansible playbooks from the GitLab repository", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "A change goes to the test server first", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "the registry team signs it off before it goes live", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "Deployments happen on Thursday evenings", "in": ["deployment_and_rollback", "known_bad_days"]},
        {"id": "F13", "quote": "we run the rollback playbook, which puts back the previous WAR file", "in": ["deployment_and_rollback"]},
        {"id": "F14", "quote": "There's test and production, plus a training copy that the registry uses for new staff", "in": ["environments"]},
        {"id": "F15", "quote": "Nagios checks the WebLogic servers and the database listener every five minutes", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "Nagios emails the infrastructure team and sends a text to the on-call person", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "The WebLogic managed servers run out of heap during results week", "in": ["common_failures"]},
        {"id": "F18", "quote": "We restart the managed servers one at a time", "in": ["common_failures"]},
        {"id": "F19", "quote": "the finance export failing when a student record has a comma in the address", "in": ["common_failures"]},
        {"id": "F20", "quote": "we fix the record and rerun the export by hand", "in": ["common_failures"]},
        {"id": "F21", "quote": "RMAN takes a full backup every Sunday and incrementals every night", "in": ["disaster_recovery"]},
        {"id": "F22", "quote": "they're copied to the second data centre", "in": ["disaster_recovery"]},
        {"id": "F23", "quote": "We can restore within six hours", "alt": ["six hours"], "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "The last restore test was in the summer", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "Staff sign in with their university account through Shibboleth", "in": ["security_controls"]},
        {"id": "F26", "quote": "Only registry managers can change a final mark", "in": ["security_controls"]},
        {"id": "F27", "quote": "every change is logged", "in": ["security_controls"]},
        {"id": "F28", "quote": "Student data must never leave the UK", "in": ["security_controls", "danger_zones"]},
        {"id": "F29", "quote": "Never run the end-of-year rollover job twice", "in": ["danger_zones"]},
        {"id": "F30", "quote": "undoing it means a database restore", "in": ["danger_zones"]},
        {"id": "F31", "quote": "Results week in June is the worst, then enrolment in September", "in": ["known_bad_days"]},
        {"id": "F32", "quote": "Avoid any change in those weeks", "in": ["known_bad_days", "deployment_and_rollback"]},
        {"id": "F33", "quote": "The student systems team in IT owns Millbrook", "in": ["ownership_escalation"]},
        {"id": "F34", "quote": "The registry owns the data", "in": ["ownership_escalation"]},
        {"id": "F35", "quote": "go to Peter Hall, the head of student systems", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "We're replacing the finance CSV export with an API", "in": ["open_responsibilities"]},
        {"id": "F37", "quote": "the Oracle licence renewal is due in December", "in": ["open_responsibilities"]},
        {"id": "F38", "quote": "get an account on the WebLogic console and read-only access to the database", "in": DAY1},
        {"id": "F39", "quote": "run a Thursday deployment yourself", "in": MONTH1},
    ],
    "covered": [s for s in ALL15 if s != "cost_optimization"],
    "missing": ["cost_optimization"],
})

KTS.append({
    "id": "d08_parcelpulse_cloudrun_noisy",
    "title": "Parcel Pulse: Cloud Run notifications, noisy speech-to-text",
    "proves": "Lower case, fillers, missing full stops: several facts run together in one sentence.",
    "transcript": """okay so um this is the handover for parcel pulse. parcel pulse is the service that sends delivery notifications to customers by sms and email and we send about two hundred thousand messages a day mostly in the evening.

so the architecture uh it runs on Cloud Run in google cloud the main service is written in Go and it reads delivery events from a Pub/Sub topic that the warehouse system publishes to. templates are stored in Firestore and the actual sending goes through Twilio for sms and SendGrid for email. we also write every message we sent to BigQuery for reporting.

deployments um we use GitHub Actions it builds the container and pushes it to Artifact Registry and then deploys a new revision. if it breaks you just roll back to the previous revision in the Cloud Run console it takes like a minute.

we have a dev project and a prod project and dev sends everything to a test phone number so you don't spam real customers.

monitoring there's a Cloud Monitoring dashboard with messages sent and the error rate and um there's an alert when the Pub/Sub subscription has more than ten thousand undelivered messages that goes to the on call pager.

so the main problems. Twilio rate limits us when there's a big batch like on black friday and messages get delayed. when that happens you lower the concurrency on the service and it drains slowly. the other one is SendGrid marking us as spam if a template has a broken link uh you fix the template and ask SendGrid support to clear the block.

backups uh Firestore has a daily export to Cloud Storage and BigQuery keeps seven days of time travel. honestly we haven't tested restoring the templates.

security wise the Twilio and SendGrid keys are in Secret Manager and the service account can only read the one Pub/Sub subscription. phone numbers are personal data so we delete message logs after ninety days.

um please don't change the Pub/Sub subscription to push mode. the service expects pull and it will drop messages.

black friday and the week before christmas are the busiest times we double the max instances before then.

cost is mostly Twilio about four thousand a month the cloud side is tiny.

it's owned by the customer comms team. if you need help escalate to Nina Patel she's the engineering manager.

stuff in progress we're adding WhatsApp as a channel and that's maybe a third done.

uh first day get access to the prod project and the Twilio console. in the first month take over the black friday preparation.""",
    "facts": [
        {"id": "F01", "quote": "parcel pulse is the service that sends delivery notifications to customers by sms and email", "in": ["system_overview"]},
        {"id": "F02", "quote": "we send about two hundred thousand messages a day", "in": ["system_overview"]},
        {"id": "F03", "quote": "it runs on Cloud Run in google cloud", "in": AR},
        {"id": "F04", "quote": "the main service is written in Go", "in": AR},
        {"id": "F05", "quote": "it reads delivery events from a Pub/Sub topic that the warehouse system publishes to", "in": AR},
        {"id": "F06", "quote": "templates are stored in Firestore", "in": AR},
        {"id": "F07", "quote": "the actual sending goes through Twilio for sms and SendGrid for email", "in": AR},
        {"id": "F08", "quote": "we also write every message we sent to BigQuery for reporting", "in": AR},
        {"id": "F09", "quote": "it builds the container and pushes it to Artifact Registry", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "roll back to the previous revision in the Cloud Run console", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "we have a dev project and a prod project", "in": ["environments"]},
        {"id": "F12", "quote": "dev sends everything to a test phone number", "in": ["environments"]},
        {"id": "F13", "quote": "there's a Cloud Monitoring dashboard with messages sent and the error rate", "in": ["monitoring_observability"]},
        {"id": "F14", "quote": "there's an alert when the Pub/Sub subscription has more than ten thousand undelivered messages", "in": ["monitoring_observability"]},
        {"id": "F15", "quote": "Twilio rate limits us when there's a big batch", "in": ["common_failures"]},
        {"id": "F16", "quote": "you lower the concurrency on the service and it drains slowly", "in": ["common_failures"]},
        {"id": "F17", "quote": "SendGrid marking us as spam if a template has a broken link", "in": ["common_failures"]},
        {"id": "F18", "quote": "ask SendGrid support to clear the block", "in": ["common_failures"]},
        {"id": "F19", "quote": "Firestore has a daily export to Cloud Storage", "in": ["disaster_recovery"]},
        {"id": "F20", "quote": "BigQuery keeps seven days of time travel", "in": ["disaster_recovery"]},
        {"id": "F21", "quote": "we haven't tested restoring the templates", "in": ["disaster_recovery"]},
        {"id": "F22", "quote": "the Twilio and SendGrid keys are in Secret Manager", "in": ["security_controls"]},
        {"id": "F23", "quote": "the service account can only read the one Pub/Sub subscription", "in": ["security_controls"]},
        {"id": "F24", "quote": "we delete message logs after ninety days", "in": ["security_controls"]},
        {"id": "F25", "quote": "don't change the Pub/Sub subscription to push mode", "in": ["danger_zones"]},
        {"id": "F26", "quote": "black friday and the week before christmas are the busiest times", "in": ["known_bad_days"]},
        {"id": "F27", "quote": "we double the max instances before then", "in": ["known_bad_days", "common_failures"]},
        {"id": "F28", "quote": "cost is mostly Twilio about four thousand a month", "in": ["cost_optimization"]},
        {"id": "F29", "quote": "the cloud side is tiny", "in": ["cost_optimization"]},
        {"id": "F30", "quote": "it's owned by the customer comms team", "in": ["ownership_escalation"]},
        {"id": "F31", "quote": "escalate to Nina Patel she's the engineering manager", "in": ["ownership_escalation"]},
        {"id": "F32", "quote": "we're adding WhatsApp as a channel", "in": ["open_responsibilities"]},
        {"id": "F33", "quote": "get access to the prod project and the Twilio console", "in": DAY1},
        {"id": "F34", "quote": "take over the black friday preparation", "in": MONTH1},
    ],
    "covered": ALL15,
    "missing": [],
})

KTS.append({
    "id": "d09_saltmarsh_azure_formal",
    "title": "Saltmarsh: Azure housing repairs, formal document read aloud",
    "proves": "Numbered section headings said aloud, passive voice, one topic per section.",
    "transcript": """Saltmarsh handover document, version two. Section one, purpose. Saltmarsh is the housing repairs system used by the council's housing department. Tenants report repairs through the website or the contact centre, and Saltmarsh schedules the work with the repair contractors. The system handles approximately one thousand two hundred repair requests per week.

Section two, architecture. The web front end is an Azure App Service running a .NET 8 application. Repair requests are stored in Azure SQL Database. Requests are sent to contractors through Azure Service Bus, and a Logic App forwards each job to the contractor's own system. Photographs uploaded by tenants are kept in Azure Blob Storage.

Section three, environments. There are three environments: development, user acceptance testing, and production. User acceptance testing is connected to a test instance of each contractor's system.

Section four, deployment. Deployments are performed through Azure DevOps release pipelines. Each release requires approval from the housing systems manager. Releases use deployment slots: the new version is deployed to the staging slot and swapped into production. Rollback is performed by swapping the slots back.

Section five, monitoring. Application Insights records requests and exceptions. An alert is raised when more than twenty Service Bus messages are moved to the dead-letter queue within an hour. Alerts are sent by email to the housing systems support mailbox.

Section six, known issues. Contractor systems occasionally reject jobs because of an unrecognised repair code. In that case, the job appears in the dead-letter queue; correct the repair code in the mapping table and resubmit the message. Additionally, the App Service can exhaust its SQL connection pool during storm events, when repair volumes spike. Scale out the App Service plan to four instances until volumes return to normal.

Section seven, recovery. Azure SQL Database provides point-in-time restore for fourteen days. The recovery point objective is one hour and the recovery time objective is eight hours. Geo-replication to the UK West region is configured but has not been tested.

Section eight, security. Staff authenticate through Microsoft Entra ID with multi-factor authentication. Tenant personal data is classified as confidential. Contractors can only see jobs assigned to them.

Section nine, restrictions. The contractor mapping table must not be edited directly in production; changes are made through the admin screen so that they are audited.

Section ten, operational calendar. Winter storms produce the highest repair volumes. The annual gas safety inspection campaign in October also doubles the number of jobs.

Section eleven, costs. The monthly Azure cost is approximately two thousand five hundred pounds. Reserved capacity for Azure SQL Database was purchased last year and reduced the database cost by a third.

Section twelve, ownership. The system is owned by the housing systems team. Escalations should be directed to Laura Bennett, Housing Systems Manager.

Section thirteen, outstanding work. The integration with the new heating contractor is in progress and is expected to complete in January.

Section fourteen, onboarding. On the first day, request access to the Azure DevOps project and the production resource group. Within the first month, the new owner should complete one release and one dead-letter queue recovery.""",
    "facts": [
        {"id": "F01", "quote": "Saltmarsh is the housing repairs system used by the council's housing department", "in": ["system_overview"]},
        {"id": "F02", "quote": "Tenants report repairs through the website or the contact centre", "in": ["system_overview"]},
        {"id": "F03", "quote": "Saltmarsh schedules the work with the repair contractors", "in": ["system_overview"]},
        {"id": "F04", "quote": "The system handles approximately one thousand two hundred repair requests per week", "in": ["system_overview"]},
        {"id": "F05", "quote": "The web front end is an Azure App Service running a .NET 8 application",
         "alt": ["The web frontend is an Azure App Service running a .NET 8 application"], "in": AR},
        {"id": "F06", "quote": "Repair requests are stored in Azure SQL Database", "in": AR},
        {"id": "F07", "quote": "Requests are sent to contractors through Azure Service Bus", "in": AR},
        {"id": "F08", "quote": "a Logic App forwards each job to the contractor's own system", "in": AR},
        {"id": "F09", "quote": "Photographs uploaded by tenants are kept in Azure Blob Storage", "in": AR},
        {"id": "F10", "quote": "There are three environments: development, user acceptance testing, and production", "in": ["environments"]},
        {"id": "F11", "quote": "User acceptance testing is connected to a test instance of each contractor's system", "in": ["environments"]},
        {"id": "F12", "quote": "Deployments are performed through Azure DevOps release pipelines", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "Each release requires approval from the housing systems manager", "in": ["deployment_and_rollback"]},
        {"id": "F14", "quote": "the new version is deployed to the staging slot and swapped into production", "in": ["deployment_and_rollback"]},
        {"id": "F15", "quote": "Rollback is performed by swapping the slots back", "in": ["deployment_and_rollback"]},
        {"id": "F16", "quote": "Application Insights records requests and exceptions", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "An alert is raised when more than twenty Service Bus messages are moved to the dead-letter queue within an hour", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "Alerts are sent by email to the housing systems support mailbox", "in": ["monitoring_observability"]},
        {"id": "F19", "quote": "Contractor systems occasionally reject jobs because of an unrecognised repair code", "in": ["common_failures"]},
        {"id": "F20", "quote": "correct the repair code in the mapping table and resubmit the message", "in": ["common_failures"]},
        {"id": "F21", "quote": "the App Service can exhaust its SQL connection pool during storm events", "in": ["common_failures"]},
        {"id": "F22", "quote": "Scale out the App Service plan to four instances until volumes return to normal", "in": ["common_failures"]},
        {"id": "F23", "quote": "Azure SQL Database provides point-in-time restore for fourteen days", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "The recovery point objective is one hour", "alt": ["one hour"], "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "the recovery time objective is eight hours", "alt": ["eight hours"], "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "Geo-replication to the UK West region is configured but has not been tested", "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "Staff authenticate through Microsoft Entra ID with multi-factor authentication", "in": ["security_controls"]},
        {"id": "F28", "quote": "Tenant personal data is classified as confidential", "in": ["security_controls"]},
        {"id": "F29", "quote": "Contractors can only see jobs assigned to them", "in": ["security_controls"]},
        {"id": "F30", "quote": "The contractor mapping table must not be edited directly in production", "in": ["danger_zones"]},
        {"id": "F31", "quote": "Winter storms produce the highest repair volumes", "in": ["known_bad_days"]},
        {"id": "F32", "quote": "The annual gas safety inspection campaign in October also doubles the number of jobs", "in": ["known_bad_days"]},
        {"id": "F33", "quote": "The monthly Azure cost is approximately two thousand five hundred pounds", "in": ["cost_optimization"]},
        {"id": "F34", "quote": "Reserved capacity for Azure SQL Database was purchased last year and reduced the database cost by a third", "in": ["cost_optimization"]},
        {"id": "F35", "quote": "The system is owned by the housing systems team", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "Escalations should be directed to Laura Bennett, Housing Systems Manager", "in": ["ownership_escalation"]},
        {"id": "F37", "quote": "The integration with the new heating contractor is in progress", "in": ["open_responsibilities"]},
        {"id": "F38", "quote": "request access to the Azure DevOps project and the production resource group", "in": DAY1},
        {"id": "F39", "quote": "the new owner should complete one release and one dead-letter queue recovery", "in": MONTH1},
    ],
    "covered": ALL15,
    "missing": [],
})

KTS.append({
    "id": "d10_pricewise_sagemaker_ramble",
    "title": "Pricewise: SageMaker pricing model, rambling with a correction",
    "proves": "Returns to a topic, asides, and a value corrected a sentence later.",
    "transcript": """Right, so Pricewise. I'll just talk through it and jump around a bit, sorry. It's the model that sets room prices for our hotels, all sixty of them, and it reprices every room every two hours. Revenue management depends on it completely, if it's wrong we either sell out too cheap or sit empty.

The model's an XGBoost model, trained on SageMaker. Training runs every Sunday as a SageMaker pipeline, and the features come from Snowflake, booking history, competitor prices, events in the city. The trained model is served from a SageMaker endpoint, and a Lambda function calls the endpoint and pushes the prices into the hotel booking system. Oh, and the competitor prices come from a scraper vendor, they drop a file in S3 every morning.

Going back to training for a second, the pipeline only promotes a new model if its error on last month's bookings is lower than the current one. Otherwise it keeps the old model.

Deploying code changes, that's GitHub Actions with the SageMaker SDK, it updates the pipeline definition and the Lambda. To roll back a model you point the endpoint at the previous model version in the model registry. That's quick.

We've got a sandbox account where you can play, and production. Sandbox has a copy of last year's bookings.

Monitoring, there's a CloudWatch dashboard for the endpoint latency and errors, and we have a check that alerts if any hotel's price moves more than thirty percent in one go. That goes to the revenue team's Slack channel and to us.

Things that go wrong. The vendor file sometimes doesn't arrive, and then the model uses stale competitor prices. If it's missing by eight, email the vendor and rerun the feature job once it lands. And sometimes the endpoint scales down overnight and the first calls in the morning time out, we set the minimum instance count to two, which sorted most of it. Sorry, I said two, it's actually three now, we raised it in the summer.

Backups. Model versions are all kept in the registry, and the training data snapshots in S3 have versioning, so we can rebuild any model. Snowflake time travel is one day only.

Security, the endpoint isn't public, it's only reachable inside the VPC, and the Snowflake user is read-only. Pricing data is commercially sensitive, so no copying it to laptops.

Really important, never set a manual price override without telling revenue management, the model will fight it every two hours.

Busy times are school holidays and the big conference weeks, we usually freeze model changes for those.

Cost, it's around three thousand dollars a month, mostly the endpoint. We tried serverless inference to save money but cold starts were too slow, so we went back.

Owned by the data science team, and revenue management are the business owners. If you're stuck, Omar Haddad, he's the head of data science.

Open stuff: we started adding weather as a feature but it's not finished, and there's an ask to support the two new hotels opening in spring.

First week, get SageMaker and Snowflake access and sit with revenue management for a morning. In your first month, run the Sunday training on your own and check the promotion decision.""",
    "facts": [
        {"id": "F01", "quote": "It's the model that sets room prices for our hotels, all sixty of them", "in": ["system_overview"]},
        {"id": "F02", "quote": "it reprices every room every two hours", "in": ["system_overview"]},
        {"id": "F03", "quote": "Revenue management depends on it completely", "in": ["system_overview"]},
        {"id": "F04", "quote": "The model's an XGBoost model, trained on SageMaker", "in": AR},
        {"id": "F05", "quote": "Training runs every Sunday as a SageMaker pipeline", "in": AR},
        {"id": "F06", "quote": "the features come from Snowflake, booking history, competitor prices, events in the city", "in": AR},
        {"id": "F07", "quote": "The trained model is served from a SageMaker endpoint", "in": AR},
        {"id": "F08", "quote": "a Lambda function calls the endpoint and pushes the prices into the hotel booking system", "in": AR},
        {"id": "F09", "quote": "the competitor prices come from a scraper vendor, they drop a file in S3 every morning", "in": AR},
        {"id": "F10", "quote": "the pipeline only promotes a new model if its error on last month's bookings is lower than the current one", "in": ["deployment_and_rollback", "architecture_reference"]},
        {"id": "F11", "quote": "GitHub Actions with the SageMaker SDK, it updates the pipeline definition and the Lambda", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "To roll back a model you point the endpoint at the previous model version in the model registry", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "We've got a sandbox account where you can play, and production", "in": ["environments"]},
        {"id": "F14", "quote": "Sandbox has a copy of last year's bookings", "in": ["environments"]},
        {"id": "F15", "quote": "there's a CloudWatch dashboard for the endpoint latency and errors", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "a check that alerts if any hotel's price moves more than thirty percent in one go", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "That goes to the revenue team's Slack channel and to us", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "The vendor file sometimes doesn't arrive, and then the model uses stale competitor prices", "in": ["common_failures"]},
        {"id": "F19", "quote": "email the vendor and rerun the feature job once it lands", "in": ["common_failures"]},
        {"id": "F20", "quote": "the endpoint scales down overnight and the first calls in the morning time out", "in": ["common_failures"]},
        {"id": "F21", "quote": "it's actually three now", "in": ["common_failures"]},
        {"id": "F22", "quote": "Model versions are all kept in the registry", "in": ["disaster_recovery"]},
        {"id": "F23", "quote": "the training data snapshots in S3 have versioning, so we can rebuild any model", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "Snowflake time travel is one day only", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "the endpoint isn't public, it's only reachable inside the VPC", "in": ["security_controls"]},
        {"id": "F26", "quote": "the Snowflake user is read-only", "in": ["security_controls"]},
        {"id": "F27", "quote": "Pricing data is commercially sensitive, so no copying it to laptops", "in": ["security_controls"]},
        {"id": "F28", "quote": "never set a manual price override without telling revenue management", "in": ["danger_zones"]},
        {"id": "F29", "quote": "Busy times are school holidays and the big conference weeks", "in": ["known_bad_days"]},
        {"id": "F30", "quote": "we usually freeze model changes for those", "in": ["known_bad_days", "deployment_and_rollback"]},
        {"id": "F31", "quote": "it's around three thousand dollars a month, mostly the endpoint", "in": ["cost_optimization"]},
        {"id": "F32", "quote": "We tried serverless inference to save money but cold starts were too slow", "in": ["cost_optimization"]},
        {"id": "F33", "quote": "Owned by the data science team", "in": ["ownership_escalation"]},
        {"id": "F34", "quote": "revenue management are the business owners", "in": ["ownership_escalation"]},
        {"id": "F35", "quote": "Omar Haddad, he's the head of data science", "in": ["ownership_escalation"]},
        {"id": "F36", "quote": "we started adding weather as a feature but it's not finished", "in": ["open_responsibilities"]},
        {"id": "F37", "quote": "an ask to support the two new hotels opening in spring", "in": ["open_responsibilities"]},
        {"id": "F38", "quote": "get SageMaker and Snowflake access and sit with revenue management for a morning", "in": DAY1},
        {"id": "F39", "quote": "run the Sunday training on your own and check the promotion decision", "in": MONTH1},
    ],
    "covered": ALL15,
    "missing": [],
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
        kt["holdout"] = False
        with open(os.path.join(out_dir, kt["id"] + ".json"), "w", encoding="utf-8") as fh:
            json.dump(kt, fh, indent=1, ensure_ascii=False)
    print(len(KTS), "KTs,", sum(len(k["facts"]) for k in KTS), "facts,", bad, "bad quotes")


if __name__ == "__main__":
    main(sys.argv[1])
