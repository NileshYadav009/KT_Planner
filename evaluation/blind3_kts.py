"""Round-7 BLIND KTs (written 25 Oct 2026, after the §9aaz fixes and before
the accuracy work that followed, before any result on them was seen). Score
them ONLY in aggregate (evaluation/accuracy.py LABEL blind3); never print or
read their per-fact results, so they stay unseen.

Four styles the other sets have little of: a question-and-answer KT with
speaker labels, a noisy recording-style KT with fillers and restarts, a
formal document read aloud, and a rambling KT with a correction mid-sentence.
Synthetic, written by the same author as the fixes, so not independent of
them; an honest check against overfitting the visible sets, not a field test.

    python blind3_kts.py OUTDIR
"""
import json
import os
import re
import sys

S = "written for round 7 (25 Oct), blind (synthetic)"
AR = ["architecture_reference", "system_overview"]
ALL_DISCUSSED = ["system_overview", "architecture_reference", "environments", "deployment_and_rollback",
                 "monitoring_observability", "common_failures", "disaster_recovery", "security_controls",
                 "danger_zones", "cost_optimization", "known_bad_days", "ownership_escalation",
                 "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(t).lower())).strip()


KTS = []

KTS.append({
    "id": "b04_harbourline_onprem_qa",
    "title": "Harbourline: on-premises container booking system, question and answer",
    "proves": "Speaker labels, a taker who asks one question per topic, on-premises stack (OpenShift, Oracle, IBM MQ).",
    "transcript": """Priya: Can you start with what Harbourline does?
Marek: Sure. Harbourline is the booking system our customer service agents use to reserve container slots on vessels. It handles roughly three thousand bookings a day, and if it is down the agents fall back to email and spreadsheets, which gets messy fast.
Priya: Where does it run?
Marek: Everything is on premises. The booking service and the pricing service are Java Spring Boot apps on our OpenShift cluster in the Rotterdam data centre. They store bookings in an Oracle database, and they talk to the vessel planning system over IBM MQ.
Priya: And how do releases work?
Marek: Jenkins builds the images and pushes them to the internal Harbor registry. Then we run the Ansible playbook that updates the OpenShift deployment. Releases need a change ticket approved by the change advisory board, which meets on Tuesdays. If a release goes wrong, we roll back with the same playbook pointing at the previous image tag.
Priya: Which environments are there?
Marek: There's a test environment and production. Test shares the Oracle server with the reporting team, so don't run load tests there without asking them.
Priya: How do you know something is wrong?
Marek: Dynatrace watches both services, and alerts go to the operations team through ServiceNow. There's also a Grafana board for the MQ queue depths.
Priya: What breaks most often?
Marek: The MQ channel to vessel planning stops after their weekly maintenance on Saturday night. Bookings queue up and nothing reaches them. You restart the channel from the MQ Explorer and the backlog drains in about ten minutes. The other one is Oracle running out of sessions when the pricing batch overlaps with office hours; we kill the batch and rerun it in the evening.
Priya: Backups?
Marek: The DBA team takes a full Oracle backup every night and keeps it for thirty days. Our recovery target is four hours, but honestly nobody has tested a full restore in the last two years.
Priya: Security?
Marek: Agents log in through Active Directory, and the services use a service account whose password is stored in CyberArk.
Priya: Anything I must never do?
Marek: Never purge the MQ queues in production. Those messages are bookings, and once they're gone we can't rebuild them.
Priya: Who owns what?
Marek: Our team owns the booking and pricing services. The infrastructure team owns OpenShift and the DBA team owns Oracle. If production is down at night, call the operations duty manager, and then me.
Priya: Anything unfinished?
Marek: The move from Oracle to PostgreSQL has been approved but not started.
Priya: What should I do first?
Marek: Get your Active Directory groups for OpenShift and Jenkins on day one, and in your first month do one release with me watching.""",
    "facts": [
        {"id": "F01", "quote": "the booking system our customer service agents use to reserve container slots on vessels", "in": ["system_overview"]},
        {"id": "F02", "quote": "roughly three thousand bookings a day", "in": ["system_overview"]},
        {"id": "F03", "quote": "if it is down the agents fall back to email and spreadsheets", "in": ["system_overview"]},
        {"id": "F04", "quote": "The booking service and the pricing service are Java Spring Boot apps on our OpenShift cluster in the Rotterdam data centre", "in": AR},
        {"id": "F05", "quote": "They store bookings in an Oracle database", "in": AR},
        {"id": "F06", "quote": "they talk to the vessel planning system over IBM MQ", "in": AR},
        {"id": "F07", "quote": "Jenkins builds the images and pushes them to the internal Harbor registry", "in": ["deployment_and_rollback", "architecture_reference"]},
        {"id": "F08", "quote": "we run the Ansible playbook that updates the OpenShift deployment", "in": ["deployment_and_rollback"]},
        {"id": "F09", "quote": "Releases need a change ticket approved by the change advisory board, which meets on Tuesdays", "in": ["deployment_and_rollback", "known_bad_days"]},
        {"id": "F10", "quote": "we roll back with the same playbook pointing at the previous image tag", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "There's a test environment and production", "in": ["environments"]},
        {"id": "F12", "quote": "Test shares the Oracle server with the reporting team, so don't run load tests there without asking them", "in": ["environments", "danger_zones"]},
        {"id": "F13", "quote": "Dynatrace watches both services", "in": ["monitoring_observability"]},
        {"id": "F14", "quote": "alerts go to the operations team through ServiceNow", "in": ["monitoring_observability", "ownership_escalation"]},
        {"id": "F15", "quote": "a Grafana board for the MQ queue depths", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "The MQ channel to vessel planning stops after their weekly maintenance on Saturday night", "in": ["common_failures", "known_bad_days"]},
        {"id": "F17", "quote": "You restart the channel from the MQ Explorer and the backlog drains in about ten minutes", "in": ["common_failures"]},
        {"id": "F18", "quote": "Oracle running out of sessions when the pricing batch overlaps with office hours", "in": ["common_failures"]},
        {"id": "F19", "quote": "we kill the batch and rerun it in the evening", "in": ["common_failures"]},
        {"id": "F20", "quote": "The DBA team takes a full Oracle backup every night and keeps it for thirty days", "in": ["disaster_recovery"]},
        {"id": "F21", "quote": "Our recovery target is four hours", "in": ["disaster_recovery"]},
        {"id": "F22", "quote": "nobody has tested a full restore in the last two years", "in": ["disaster_recovery"]},
        {"id": "F23", "quote": "Agents log in through Active Directory", "in": ["security_controls"]},
        {"id": "F24", "quote": "the services use a service account whose password is stored in CyberArk", "in": ["security_controls"]},
        {"id": "F25", "quote": "Never purge the MQ queues in production", "in": ["danger_zones"]},
        {"id": "F26", "quote": "Our team owns the booking and pricing services", "in": ["ownership_escalation"]},
        {"id": "F27", "quote": "The infrastructure team owns OpenShift and the DBA team owns Oracle", "in": ["ownership_escalation"]},
        {"id": "F28", "quote": "If production is down at night, call the operations duty manager, and then me", "in": ["ownership_escalation"]},
        {"id": "F29", "quote": "The move from Oracle to PostgreSQL has been approved but not started", "in": ["open_responsibilities"]},
        {"id": "F30", "quote": "Get your Active Directory groups for OpenShift and Jenkins on day one", "in": ["day1_survival_checklist"]},
        {"id": "F31", "quote": "in your first month do one release with me watching", "in": ["first_30_day_ownership"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback",
                "monitoring_observability", "common_failures", "disaster_recovery", "security_controls", "danger_zones",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": ["cost_optimization"],
})

KTS.append({
    "id": "b05_ridepoint_cloudrun_noisy",
    "title": "Ridepoint payouts: GCP Cloud Run, recording-style speech with fillers and restarts",
    "proves": "Fillers, repeated words and restarts as speech-to-text leaves them; a guess about RTO/RPO; a never-do said as 'really important'.",
    "transcript": """Okay so um, this is Ridepoint payouts. It's the service that pays our drivers every week, uh, about forty thousand drivers in three countries. If payouts are late, drivers notice within an hour and support gets flooded.
So the, the main thing is a Go service on Cloud Run. It reads the completed trips from Pub/Sub, works out what each driver earned, and writes the balances into Cloud SQL, which is Postgres. On Thursday night a scheduled job, it's Cloud Scheduler, triggers the payout run and we send the transfers to Adyen.
Deploys, um, Cloud Build builds on every merge and deploys to staging by itself. For prod you promote the same image with the gcloud command in the runbook, and if it's bad you just send traffic back to the previous revision, which takes a minute.
We've got staging and prod. Staging talks to the Adyen test environment so no real money moves there.
Monitoring is Cloud Monitoring and we, uh, we have alerts in Opsgenie for the payout run failing and for the Pub/Sub backlog growing.
Things that go wrong. Adyen rate limits us when the run starts, you'll see four twenty nines in the logs. The job retries by itself, just let it finish, don't start a second run. And sometimes a driver's bank details are invalid and that payout fails, those go to the finance ops queue and they fix it by hand.
Backups, Cloud SQL has point in time recovery switched on, seven days. We haven't written down an RPO or RTO, I'd guess a few hours but don't quote me on that.
Bank details are encrypted with Cloud KMS and only the payout service account can decrypt them.
Really important: never rerun a payout run for the same week. It pays the drivers twice and getting money back is a nightmare.
Cost wise it's cheap, the only thing is Cloud SQL, we sized it down last quarter.
Thursday night and the first payout after a public holiday are the busy ones, avoid deploying then.
It's our team, payments platform. For Adyen problems contact their support through the merchant portal. If the run fails at night, page the on-call and then Lucia, she's the team lead.
There's still a half finished migration to Adyen's new payout API, the new client is written but not switched on.
First week get access to the GCP project and Opsgenie. By the end of the month you should run a payout week end to end with someone watching.""",
    "facts": [
        {"id": "F01", "quote": "the service that pays our drivers every week", "in": ["system_overview"]},
        {"id": "F02", "quote": "about forty thousand drivers in three countries", "in": ["system_overview"]},
        {"id": "F03", "quote": "If payouts are late, drivers notice within an hour and support gets flooded", "in": ["system_overview"]},
        {"id": "F04", "quote": "a Go service on Cloud Run", "in": AR},
        {"id": "F05", "quote": "It reads the completed trips from Pub/Sub, works out what each driver earned, and writes the balances into Cloud SQL", "in": AR},
        {"id": "F06", "quote": "triggers the payout run", "in": AR + ["known_bad_days"]},
        {"id": "F07", "quote": "we send the transfers to Adyen", "in": AR},
        {"id": "F08", "quote": "Cloud Build builds on every merge and deploys to staging by itself", "in": ["deployment_and_rollback"]},
        {"id": "F09", "quote": "For prod you promote the same image with the gcloud command in the runbook", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "if it's bad you just send traffic back to the previous revision, which takes a minute", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "We've got staging and prod", "in": ["environments"]},
        {"id": "F12", "quote": "Staging talks to the Adyen test environment so no real money moves there", "in": ["environments"]},
        {"id": "F13", "quote": "Monitoring is Cloud Monitoring", "in": ["monitoring_observability"]},
        {"id": "F14", "quote": "alerts in Opsgenie for the payout run failing and for the Pub/Sub backlog growing", "in": ["monitoring_observability"]},
        {"id": "F15", "quote": "Adyen rate limits us when the run starts", "in": ["common_failures"]},
        {"id": "F16", "quote": "The job retries by itself, just let it finish, don't start a second run", "in": ["common_failures", "danger_zones"]},
        {"id": "F17", "quote": "a driver's bank details are invalid and that payout fails", "in": ["common_failures"]},
        {"id": "F18", "quote": "those go to the finance ops queue and they fix it by hand", "in": ["common_failures", "ownership_escalation"]},
        {"id": "F19", "quote": "Cloud SQL has point in time recovery switched on, seven days", "in": ["disaster_recovery"]},
        {"id": "F20", "quote": "We haven't written down an RPO or RTO", "in": ["disaster_recovery"]},
        {"id": "F21", "quote": "Bank details are encrypted with Cloud KMS and only the payout service account can decrypt them", "in": ["security_controls"]},
        {"id": "F22", "quote": "never rerun a payout run for the same week", "in": ["danger_zones"]},
        {"id": "F23", "quote": "It pays the drivers twice and getting money back is a nightmare", "in": ["danger_zones"]},
        {"id": "F24", "quote": "the only thing is Cloud SQL, we sized it down last quarter", "in": ["cost_optimization"]},
        {"id": "F25", "quote": "Thursday night and the first payout after a public holiday are the busy ones, avoid deploying then", "in": ["known_bad_days"]},
        {"id": "F26", "quote": "It's our team, payments platform", "in": ["ownership_escalation"]},
        {"id": "F27", "quote": "For Adyen problems contact their support through the merchant portal", "in": ["ownership_escalation"]},
        {"id": "F28", "quote": "If the run fails at night, page the on-call and then Lucia, she's the team lead", "in": ["ownership_escalation"]},
        {"id": "F29", "quote": "the new client is written but not switched on", "in": ["open_responsibilities"]},
        {"id": "F30", "quote": "First week get access to the GCP project and Opsgenie", "in": ["day1_survival_checklist"]},
        {"id": "F31", "quote": "By the end of the month you should run a payout week end to end with someone watching", "in": ["first_30_day_ownership"]},
    ],
    "covered": ALL_DISCUSSED,
    "missing": [],
})

KTS.append({
    "id": "b06_claimsway_azure_formal",
    "title": "Claimsway: Azure claims intake platform, a formal document read aloud",
    "proves": "Passive, formal sentences with no speaker; every section said once, in order.",
    "transcript": """This document describes Claimsway, the claims intake platform for the motor insurance division. Claimsway receives first notification of loss reports from the customer web form and the call centre, validates them and creates a claim in the core policy system. Approximately two thousand five hundred claims are received per day, rising sharply after storms.

The platform consists of Azure Functions written in C#. Incoming reports are placed on an Azure Service Bus queue. Claim documents and photographs are stored in Azure Blob Storage. Claim state is held in Azure Cosmos DB. The web form is published through Azure Application Gateway with the web application firewall enabled.

Infrastructure is defined in Bicep. GitHub Actions deploys to the test environment on every merge to the main branch. Deployment to production requires approval from the claims product owner in GitHub. Rollback is performed by redeploying the previous release from the GitHub Actions history.

There are three environments: development, test and production. The test environment uses masked copies of production claims.

Application Insights collects logs and traces. Alerts are sent to the claims support channel in Microsoft Teams.

The most frequent incident is a poison message on the Service Bus queue after the core policy system rejects a claim. Such messages are moved to the dead-letter queue and must be resubmitted from Service Bus Explorer after the policy data is corrected. A second known issue is Cosmos DB throttling during storm peaks; the provisioned throughput is raised temporarily by the on-call engineer.

Cosmos DB continuous backup is enabled with a thirty day restore window. Blob Storage uses geo-redundant storage. The recovery time objective is eight hours. A disaster recovery test has not yet been performed.

All personal data is encrypted at rest. Access to production is granted through Privileged Identity Management for a maximum of four hours.

Claim photographs must never be deleted manually, because they are required as legal evidence for seven years.

Storage of photographs is the largest cost; a lifecycle policy moves photographs older than ninety days to the archive tier.

After major storms, claim volumes can triple within a day.

The claims platform team owns Claimsway. The core policy system is owned by the policy administration team. Escalation for production incidents is to the claims platform manager, Sanjay Mehta.

The integration with the new fraud scoring service remains outstanding.

On the first day, the new owner should request Privileged Identity Management eligibility for the production subscription. Within the first thirty days, the new owner should complete a dead-letter resubmission under supervision.""",
    "facts": [
        {"id": "F01", "quote": "the claims intake platform for the motor insurance division", "in": ["system_overview"]},
        {"id": "F02", "quote": "receives first notification of loss reports from the customer web form and the call centre", "in": AR},
        {"id": "F03", "quote": "Approximately two thousand five hundred claims are received per day", "in": ["system_overview"]},
        {"id": "F04", "quote": "The platform consists of Azure Functions written in C#", "in": AR},
        {"id": "F05", "quote": "Incoming reports are placed on an Azure Service Bus queue", "in": AR},
        {"id": "F06", "quote": "Claim documents and photographs are stored in Azure Blob Storage", "in": AR},
        {"id": "F07", "quote": "Claim state is held in Azure Cosmos DB", "in": AR},
        {"id": "F08", "quote": "The web form is published through Azure Application Gateway with the web application firewall enabled", "in": AR + ["security_controls"]},
        {"id": "F09", "quote": "Infrastructure is defined in Bicep", "in": ["architecture_reference", "deployment_and_rollback", "environments"]},
        {"id": "F10", "quote": "GitHub Actions deploys to the test environment on every merge to the main branch", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "Deployment to production requires approval from the claims product owner in GitHub", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "Rollback is performed by redeploying the previous release from the GitHub Actions history", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "There are three environments: development, test and production", "in": ["environments"]},
        {"id": "F14", "quote": "The test environment uses masked copies of production claims", "in": ["environments", "security_controls"]},
        {"id": "F15", "quote": "Application Insights collects logs and traces", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "Alerts are sent to the claims support channel in Microsoft Teams", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "a poison message on the Service Bus queue after the core policy system rejects a claim", "in": ["common_failures"]},
        {"id": "F18", "quote": "must be resubmitted from Service Bus Explorer after the policy data is corrected", "in": ["common_failures"]},
        {"id": "F19", "quote": "Cosmos DB throttling during storm peaks", "in": ["common_failures"]},
        {"id": "F20", "quote": "the provisioned throughput is raised temporarily by the on-call engineer", "in": ["common_failures"]},
        {"id": "F21", "quote": "Cosmos DB continuous backup is enabled with a thirty day restore window", "in": ["disaster_recovery"]},
        {"id": "F22", "quote": "Blob Storage uses geo-redundant storage", "in": ["disaster_recovery"]},
        {"id": "F23", "quote": "is eight hours", "terms": ["eight hours"], "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "test has not yet been performed", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "All personal data is encrypted at rest", "in": ["security_controls"]},
        {"id": "F26", "quote": "Access to production is granted through Privileged Identity Management for a maximum of four hours", "in": ["security_controls"]},
        {"id": "F27", "quote": "Claim photographs must never be deleted manually, because they are required as legal evidence for seven years", "in": ["danger_zones"]},
        {"id": "F28", "quote": "Storage of photographs is the largest cost", "in": ["cost_optimization"]},
        {"id": "F29", "quote": "a lifecycle policy moves photographs older than ninety days to the archive tier", "in": ["cost_optimization"]},
        {"id": "F30", "quote": "After major storms, claim volumes can triple within a day", "in": ["known_bad_days"]},
        {"id": "F31", "quote": "The claims platform team owns Claimsway", "in": ["ownership_escalation"]},
        {"id": "F32", "quote": "The core policy system is owned by the policy administration team", "in": ["ownership_escalation"]},
        {"id": "F33", "quote": "Escalation for production incidents is to the claims platform manager, Sanjay Mehta", "in": ["ownership_escalation"]},
        {"id": "F34", "quote": "The integration with the new fraud scoring service remains outstanding", "in": ["open_responsibilities"]},
        {"id": "F35", "quote": "request Privileged Identity Management eligibility for the production subscription", "in": ["day1_survival_checklist"]},
        {"id": "F36", "quote": "complete a dead-letter resubmission under supervision", "in": ["first_30_day_ownership"]},
    ],
    "covered": ALL_DISCUSSED,
    "missing": [],
})

KTS.append({
    "id": "b07_tasteful_sagemaker_ramble",
    "title": "Tasteful: AWS SageMaker recommendations, a rambling KT with a correction",
    "proves": "Run-on sentences, a correction mid-sentence ('every week, sorry, every night now'), an unknown fix, an idea that is not yet work.",
    "transcript": """Right, so Tasteful is the recommendation service for the recipe app, it decides which recipes you see on the home screen. Something like two million people open the app a week, and if recommendations break the app falls back to the most popular recipes, so it's not an outage, but engagement drops.
How it works, there's a feature service, Python on EKS, that reads user history from Redis and calls a SageMaker endpoint for the model. Training data lands in S3 every night from the events pipeline. Airflow, the managed one on AWS, MWAA, runs the training. We retrain every week, sorry, every night now, we changed that in March.
The model is deployed by an Airflow task that registers it and updates the SageMaker endpoint, but only if the offline metrics beat the current model. The feature service itself goes out through GitHub Actions and Argo CD. If a new model is bad, you point the endpoint back at the previous model version in the registry.
Infra is all Terraform, one workspace per environment. We have dev and prod, there's no staging for the model, dev uses a ten percent sample of the data.
Datadog has the dashboards. The alert I care about is the click-through rate dropping more than twenty percent day on day, that goes to Slack, not paging.
What breaks, the nightly training fails when the events pipeline is late, then the old model just stays, which is fine for a day or two. Rerun the DAG once the data is there. The feature service sometimes times out calling SageMaker during the evening peak, we added autoscaling on the endpoint but it still happens now and then; I don't know the proper fix yet.
Backups, the models are versioned in the SageMaker model registry and the training data stays in S3 for ninety days. If Redis is lost the features rebuild from S3 overnight.
The endpoint isn't public, the feature service calls it through a VPC endpoint, and user ids are hashed before they go into training data.
Never train on unhashed user data, legal were very clear about that.
SageMaker endpoints are the big bill. We use a spot training job, which halved the training cost.
Evenings between six and nine are the peak, and Sunday is the biggest day for cooking, so we don't release on Sundays.
I own it with one other engineer, Mo. The data platform team owns the events pipeline. If recommendations are down for real, ping Mo, and then the head of data, Grace.
Open stuff: we want to add an A/B testing framework but it's only an idea right now, no ticket.
Day one, get the AWS role and Datadog. In your first month, ship one model change through the whole pipeline.""",
    "facts": [
        {"id": "F01", "quote": "it decides which recipes you see on the home screen", "in": ["system_overview"]},
        {"id": "F02", "quote": "two million people open the app a week", "in": ["system_overview"]},
        {"id": "F03", "quote": "the app falls back to the most popular recipes, so it's not an outage, but engagement drops", "in": ["system_overview"]},
        {"id": "F04", "quote": "reads user history from Redis and calls a SageMaker endpoint for the model", "in": AR},
        {"id": "F05", "quote": "Training data lands in S3 every night from the events pipeline", "in": AR},
        {"id": "F06", "quote": "runs the training", "terms": ["MWAA", "runs the training"], "in": AR},
        {"id": "F07", "quote": "every night now, we changed that in March", "in": AR + ["deployment_and_rollback"]},
        {"id": "F08", "quote": "The model is deployed by an Airflow task that registers it and updates the SageMaker endpoint, but only if the offline metrics beat the current model", "in": ["deployment_and_rollback"]},
        {"id": "F09", "quote": "The feature service itself goes out through GitHub Actions", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "If a new model is bad, you point the endpoint back at the previous model version in the registry", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "Infra is all Terraform, one workspace per environment", "in": ["environments", "architecture_reference", "deployment_and_rollback"]},
        {"id": "F12", "quote": "there's no staging for the model", "in": ["environments"]},
        {"id": "F13", "quote": "dev uses a ten percent sample of the data", "in": ["environments"]},
        {"id": "F14", "quote": "Datadog has the dashboards", "in": ["monitoring_observability"]},
        {"id": "F15", "quote": "the click-through rate dropping more than twenty percent day on day, that goes to Slack, not paging", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "the nightly training fails when the events pipeline is late", "in": ["common_failures"]},
        {"id": "F17", "quote": "Rerun the DAG once the data is there", "in": ["common_failures"]},
        {"id": "F18", "quote": "The feature service sometimes times out calling SageMaker during the evening peak", "in": ["common_failures"]},
        {"id": "F19", "quote": "I don't know the proper fix yet", "in": ["common_failures"]},
        {"id": "F20", "quote": "the models are versioned in the SageMaker model registry and the training data stays in S3 for ninety days", "in": ["disaster_recovery"]},
        {"id": "F21", "quote": "If Redis is lost the features rebuild from S3 overnight", "in": ["disaster_recovery"]},
        {"id": "F22", "quote": "the feature service calls it through a VPC endpoint", "in": ["security_controls", "architecture_reference"]},
        {"id": "F23", "quote": "user ids are hashed before they go into training data", "in": ["security_controls"]},
        {"id": "F24", "quote": "Never train on unhashed user data", "in": ["danger_zones"]},
        {"id": "F25", "quote": "SageMaker endpoints are the big bill", "in": ["cost_optimization"]},
        {"id": "F26", "quote": "We use a spot training job, which halved the training cost", "in": ["cost_optimization"]},
        {"id": "F27", "quote": "Evenings between six and nine are the peak", "in": ["known_bad_days"]},
        {"id": "F28", "quote": "Sunday is the biggest day for cooking, so we don't release on Sundays", "in": ["known_bad_days"]},
        {"id": "F29", "quote": "I own it with one other engineer, Mo", "in": ["ownership_escalation"]},
        {"id": "F30", "quote": "The data platform team owns the events pipeline", "in": ["ownership_escalation"]},
        {"id": "F31", "quote": "If recommendations are down for real, ping Mo, and then the head of data, Grace", "in": ["ownership_escalation"]},
        {"id": "F32", "quote": "we want to add an A/B testing framework but it's only an idea right now, no ticket", "in": ["open_responsibilities"]},
        {"id": "F33", "quote": "get the AWS role and Datadog", "in": ["day1_survival_checklist"]},
        {"id": "F34", "quote": "ship one model change through the whole pipeline", "in": ["first_30_day_ownership"]},
    ],
    "covered": ALL_DISCUSSED,
    "missing": [],
})


def main(out_dir):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from devops_transcription import clean_transcript
    os.makedirs(out_dir, exist_ok=True)
    bad = 0
    for kt in KTS:
        text = norm(clean_transcript(kt["transcript"]))
        for f in kt["facts"]:
            if not any(norm(q) in text for q in [f["quote"]] + f.get("alt", [])):
                print("QUOTE NOT IN CLEANED TRANSCRIPT", kt["id"], f["id"])
                bad += 1
        kt["source"] = S
        kt["holdout"] = True
        with open(os.path.join(out_dir, kt["id"] + ".json"), "w", encoding="utf-8") as fh:
            json.dump(kt, fh, indent=1, ensure_ascii=False)
    print(len(KTS), "KTs,", sum(len(k["facts"]) for k in KTS), "facts,", bad, "quotes to fix")


if __name__ == "__main__":
    main(sys.argv[1])
