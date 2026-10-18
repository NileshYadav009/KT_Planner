"""Round-6 BLIND KTs (written 17 Oct 2026 after the round-6 changes were
fixed in code, before any result on them was seen). Score them ONLY in
aggregate (evaluation/accuracy.py LABEL blind2); never print or read their per-fact results,
so they stay unseen for the next round.

    python blind_kts.py OUTDIR
"""
import json
import os
import re
import sys

S = "written for round 6, blind (synthetic)"
AR = ["architecture_reference", "system_overview"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(t).lower())).strip()


KTS = []

KTS.append({
    "id": "b01_findit_opensearch_walkthrough",
    "title": "FindIt: AWS OpenSearch product search, unlabelled screen-share walkthrough",
    "proves": "No speaker names, no headings or questions; topics change without being named.",
    "transcript": """So this is FindIt, the product search behind the shop's search box and the category pages. Every search a customer types goes through it, around four million queries a day, and when it's slow people just leave, so conversion drops almost immediately.

What you're looking at here is the indexing side. Product updates come from the catalogue service as events on an SNS topic, a Lambda picks them up and writes the documents into Amazon OpenSearch Service. The query API is a Kotlin service on ECS that sits behind CloudFront and an Application Load Balancer. Synonyms and boosting rules are kept in a DynamoDB table that the merchandisers edit through an admin page.

This tab is CodePipeline. A merge to main builds the container, runs the relevance tests against a sample index, and then deploys to staging. Production is a manual approval step in the same pipeline. If the new version misbehaves, you approve the previous build again and it rolls back in a few minutes.

We have dev, staging and production. Staging has a full copy of the production index, refreshed every Sunday, so the relevance numbers there are realistic.

Here's the CloudWatch dashboard. The two numbers that matter are p95 query latency and the zero-results rate. Both have alarms that go to PagerDuty, and the zero-results alarm also posts in the merchandising channel.

Okay, the thing you'll get woken up for. A full reindex runs every night at two, and if a node runs out of disk the cluster goes yellow and the reindex stops halfway. You'll see the old index still serving, so customers don't notice, but new products are missing. Free up space by deleting the oldest index snapshot from the cluster, then restart the reindex job from the Step Functions console. The other one is a merchandiser adding a synonym that's too broad, like making shoes and boots the same word, and suddenly the results look wrong. Remove the synonym in the admin page and clear the query cache.

Snapshots of the cluster go to S3 every hour and we keep fourteen days of them. In a real disaster we can rebuild the whole index from the catalogue in about six hours. We tried that once last year.

The OpenSearch domain is inside a VPC with no public endpoint. The admin page needs Okta login, and only the merchandising team has the editor role.

Please don't change the number of shards on the live index. It needs a full reindex and blue green alias switch to do safely, and getting it wrong takes search down.

OpenSearch is by far the most expensive part. We moved the staging cluster to smaller instances and turned it off at weekends, which saved about a fifth.

The week before Christmas and the January sale are when we freeze changes, nothing goes to production then.

The search team owns FindIt. For anything about ranking or synonyms talk to the merchandising lead, Ines Duarte. If production is down, page the on-call and then the engineering manager, Tomasz Nowak.

The upgrade from OpenSearch 1.3 to 2.x is half planned. There's a ticket and a test cluster, but nobody has done the dry run.

First day, ask for access to the AWS account, PagerDuty and the CloudWatch dashboard. Over the first few weeks, shadow two on-call shifts and run one reindex yourself with me watching.""",
    "facts": [
        {"id": "F01", "quote": "the product search behind the shop's search box and the category pages", "in": ["system_overview"]},
        {"id": "F02", "quote": "around four million queries a day", "in": ["system_overview"]},
        {"id": "F03", "quote": "when it's slow people just leave, so conversion drops almost immediately", "in": ["system_overview"]},
        {"id": "F04", "quote": "Product updates come from the catalogue service as events on an SNS topic", "in": AR},
        {"id": "F05", "quote": "a Lambda picks them up and writes the documents into Amazon OpenSearch Service", "in": AR},
        {"id": "F06", "quote": "The query API is a Kotlin service on ECS that sits behind CloudFront and an Application Load Balancer", "in": AR},
        {"id": "F07", "quote": "Synonyms and boosting rules are kept in a DynamoDB table", "in": AR},
        {"id": "F08", "quote": "A merge to main builds the container, runs the relevance tests against a sample index", "in": ["deployment_and_rollback"]},
        {"id": "F09", "quote": "Production is a manual approval step in the same pipeline", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "you approve the previous build again and it rolls back in a few minutes", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "We have dev, staging and production", "in": ["environments"]},
        {"id": "F12", "quote": "Staging has a full copy of the production index, refreshed every Sunday", "in": ["environments"]},
        {"id": "F13", "quote": "The two numbers that matter are p95 query latency and the zero-results rate", "in": ["monitoring_observability"]},
        {"id": "F14", "quote": "Both have alarms that go to PagerDuty", "in": ["monitoring_observability"]},
        {"id": "F15", "quote": "the zero-results alarm also posts in the merchandising channel", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "if a node runs out of disk the cluster goes yellow and the reindex stops halfway", "in": ["common_failures"]},
        {"id": "F17", "quote": "new products are missing", "in": ["common_failures"]},
        {"id": "F18", "quote": "Free up space by deleting the oldest index snapshot from the cluster", "in": ["common_failures"]},
        {"id": "F19", "quote": "restart the reindex job from the Step Functions console", "in": ["common_failures"]},
        {"id": "F20", "quote": "a merchandiser adding a synonym that's too broad", "in": ["common_failures"]},
        {"id": "F21", "quote": "Remove the synonym in the admin page and clear the query cache", "in": ["common_failures"]},
        {"id": "F22", "quote": "Snapshots of the cluster go to S3 every hour and we keep fourteen days of them", "in": ["disaster_recovery"]},
        {"id": "F23", "quote": "we can rebuild the whole index from the catalogue in about six hours", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "We tried that once last year", "in": ["disaster_recovery"]},
        {"id": "F25", "quote": "The OpenSearch domain is inside a VPC with no public endpoint", "in": ["security_controls"]},
        {"id": "F26", "quote": "The admin page needs Okta login", "in": ["security_controls"]},
        {"id": "F27", "quote": "only the merchandising team has the editor role", "in": ["security_controls"]},
        {"id": "F28", "quote": "don't change the number of shards on the live index", "in": ["danger_zones"]},
        {"id": "F29", "quote": "OpenSearch is by far the most expensive part", "in": ["cost_optimization"]},
        {"id": "F30", "quote": "We moved the staging cluster to smaller instances and turned it off at weekends", "in": ["cost_optimization"]},
        {"id": "F31", "quote": "The week before Christmas and the January sale are when we freeze changes", "in": ["known_bad_days"]},
        {"id": "F32", "quote": "The search team owns FindIt", "in": ["ownership_escalation"], "terms": ["search team", "findit"]},
        {"id": "F33", "quote": "talk to the merchandising lead, Ines Duarte", "in": ["ownership_escalation"]},
        {"id": "F34", "quote": "page the on-call and then the engineering manager, Tomasz Nowak", "in": ["ownership_escalation"]},
        {"id": "F35", "quote": "The upgrade from OpenSearch 1.3 to 2.x is half planned", "in": ["open_responsibilities"]},
        {"id": "F36", "quote": "nobody has done the dry run", "in": ["open_responsibilities"]},
        {"id": "F37", "quote": "ask for access to the AWS account, PagerDuty and the CloudWatch dashboard", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F38", "quote": "shadow two on-call shifts and run one reindex yourself with me watching", "in": ["first_30_day_ownership", "day1_survival_checklist"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "cost_optimization", "known_bad_days",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": [],
})

KTS.append({
    "id": "b02_paybridge_logicapps_document",
    "title": "PayBridge: Azure Logic Apps payroll integration, handover document read aloud in paragraphs",
    "proves": "Written-document register in paragraphs, security and compliance stated as nouns.",
    "transcript": """PayBridge transfers approved payroll data from the HR system, Workday, to the SAP payroll engine and returns payslips to employees. It serves around eleven thousand employees in four countries. A failure on payroll day means people are not paid on time, which is treated as a severity one incident.

PayBridge consists of Azure Logic Apps workflows, an Azure Service Bus namespace for buffering employee changes, and an Azure SQL database that stores the mapping between Workday and SAP employee identifiers. Payslip PDFs are written to an Azure Storage account and shown to employees through the intranet. The integration with SAP uses the on-premises data gateway installed in the Frankfurt data centre.

Deployments are made with Azure DevOps release pipelines from ARM templates kept in the PayBridge repository. Every release to production requires the approval of the payroll product owner and is performed outside the payroll window. A rollback is performed by re-running the previous release, which restores the earlier workflow definitions.

The integration has a development environment, a test environment connected to the SAP quality system, and production. The test environment contains synthetic employees only.

Application Insights and Log Analytics capture every workflow run. An alert is raised when more than five workflow runs fail within an hour, and alerts are sent to the integration support mailbox and to the on-call phone.

The most frequent incident is the on-premises data gateway losing its connection after Windows updates on the gateway server. All transfers then fail with a gateway error. Restarting the gateway service on the server resolves it. A second recurring incident is an employee record rejected by SAP because a mandatory field is empty in Workday; the failed message is moved to the dead-letter queue, and HR must correct the record before the message is resubmitted.

The Azure SQL database uses geo-replication to the paired region, and the Logic Apps definitions can be redeployed from the repository. The recovery point objective is one hour. A recovery test has not been performed since the move to the new subscription.

All secrets, including the SAP service user password, are stored in Azure Key Vault. Personal data in the payslip storage account is encrypted with customer-managed keys. Access to production is granted through Privileged Identity Management for a maximum of four hours.

The dead-letter queue must not be purged; it holds the only copy of rejected employee changes. The mapping table must not be edited manually.

Payroll runs on the twentieth of each month, and the three working days before it are a change freeze.

The Logic Apps consumption plan is billed per action. Batching employee changes into groups of fifty reduced the monthly cost by roughly forty percent.

PayBridge is owned by the HR technology team. Escalation goes first to the integration support lead, Amara Okafor, and then to the head of HR technology.

The replacement of the on-premises data gateway with a direct SAP API connection is scoped but has not been approved.

On the first day, the new owner should request Privileged Identity Management eligibility and access to the integration support mailbox. During the first month, the new owner should observe one payroll run and lead the next release with supervision.""",
    "facts": [
        {"id": "F01", "quote": "PayBridge transfers approved payroll data from the HR system, Workday, to the SAP payroll engine", "in": ["system_overview"]},
        {"id": "F02", "quote": "returns payslips to employees", "in": ["system_overview"]},
        {"id": "F03", "quote": "It serves around eleven thousand employees in four countries", "in": ["system_overview"]},
        {"id": "F04", "quote": "A failure on payroll day means people are not paid on time", "in": ["system_overview"]},
        {"id": "F05", "quote": "PayBridge consists of Azure Logic Apps workflows", "in": AR},
        {"id": "F06", "quote": "an Azure Service Bus namespace for buffering employee changes", "in": AR},
        {"id": "F07", "quote": "an Azure SQL database that stores the mapping between Workday and SAP employee identifiers", "in": AR},
        {"id": "F08", "quote": "Payslip PDFs are written to an Azure Storage account and shown to employees through the intranet", "in": AR},
        {"id": "F09", "quote": "The integration with SAP uses the on-premises data gateway installed in the Frankfurt data centre", "in": AR},
        {"id": "F10", "quote": "Deployments are made with Azure DevOps release pipelines from ARM templates kept in the PayBridge repository", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "Every release to production requires the approval of the payroll product owner", "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "A rollback is performed by re-running the previous release", "in": ["deployment_and_rollback"]},
        {"id": "F13", "quote": "a test environment connected to the SAP quality system", "in": ["environments"]},
        {"id": "F14", "quote": "The test environment contains synthetic employees only", "in": ["environments"]},
        {"id": "F15", "quote": "Application Insights and Log Analytics capture every workflow run", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "An alert is raised when more than five workflow runs fail within an hour", "in": ["monitoring_observability"]},
        {"id": "F17", "quote": "alerts are sent to the integration support mailbox and to the on-call phone", "in": ["monitoring_observability"]},
        {"id": "F18", "quote": "the on-premises data gateway losing its connection after Windows updates on the gateway server", "in": ["common_failures"]},
        {"id": "F19", "quote": "All transfers then fail with a gateway error", "in": ["common_failures"]},
        {"id": "F20", "quote": "Restarting the gateway service on the server resolves it", "in": ["common_failures"]},
        {"id": "F21", "quote": "an employee record rejected by SAP because a mandatory field is empty in Workday", "in": ["common_failures"]},
        {"id": "F22", "quote": "HR must correct the record before the message is resubmitted", "in": ["common_failures"]},
        {"id": "F23", "quote": "The Azure SQL database uses geo-replication to the paired region", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "the Logic Apps definitions can be redeployed from the repository", "in": ["disaster_recovery", "deployment_and_rollback"]},
        {"id": "F25", "quote": "The recovery point objective is one hour", "alt": ["one hour"], "in": ["disaster_recovery"]},
        {"id": "F26", "quote": "A recovery test has not been performed since the move to the new subscription", "in": ["disaster_recovery"]},
        {"id": "F27", "quote": "All secrets, including the SAP service user password, are stored in Azure Key Vault", "in": ["security_controls"]},
        {"id": "F28", "quote": "Personal data in the payslip storage account is encrypted with customer-managed keys", "in": ["security_controls"]},
        {"id": "F29", "quote": "Access to production is granted through Privileged Identity Management for a maximum of four hours", "in": ["security_controls"]},
        {"id": "F30", "quote": "The dead-letter queue must not be purged", "in": ["danger_zones"]},
        {"id": "F31", "quote": "The mapping table must not be edited manually", "in": ["danger_zones"]},
        {"id": "F32", "quote": "Payroll runs on the twentieth of each month", "in": ["known_bad_days", "system_overview"]},
        {"id": "F33", "quote": "the three working days before it are a change freeze", "in": ["known_bad_days"]},
        {"id": "F34", "quote": "The Logic Apps consumption plan is billed per action", "in": ["cost_optimization"]},
        {"id": "F35", "quote": "Batching employee changes into groups of fifty reduced the monthly cost by roughly forty percent", "in": ["cost_optimization"]},
        {"id": "F36", "quote": "PayBridge is owned by the HR technology team", "in": ["ownership_escalation"], "terms": ["hr technology team", "paybridge"]},
        {"id": "F37", "quote": "Escalation goes first to the integration support lead, Amara Okafor", "in": ["ownership_escalation"]},
        {"id": "F38", "quote": "The replacement of the on-premises data gateway with a direct SAP API connection is scoped but has not been approved", "in": ["open_responsibilities"]},
        {"id": "F39", "quote": "request Privileged Identity Management eligibility and access to the integration support mailbox", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F40", "quote": "observe one payroll run and lead the next release with supervision", "in": ["first_30_day_ownership", "handover_completion"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "known_bad_days", "cost_optimization",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": [],
})

KTS.append({
    "id": "b03_streamline_dataflow_noisy",
    "title": "Pulse: GCP Dataflow streaming analytics, noisy speech-to-text with false starts",
    "proves": "Repeated words, false starts, fillers, a topic dropped and picked up again.",
    "transcript": """right so uh pulse. pulse is the the clickstream analytics thing, it takes every click from the website and the app and gives the product managers near real time dashboards, like how many people are in checkout right now. it's not customer facing but the growth team stares at it all day so they will notice in like five minutes if it's down.

so the clicks come in through a small Go collector on Cloud Run, it publishes to Pub/Sub, and then uh a Dataflow streaming job does the sessionising and writes into BigQuery. the dashboards are Looker Studio on top of BigQuery. oh and there's a Bigtable table for the live counters because BigQuery was too slow for that.

deploying, so the Dataflow job is a Java Beam pipeline and we deploy it with a Cloud Build trigger, it does an update in place of the running job, um, unless the change isn't compatible, then you have to drain the old job and start a new one. the collector is just a normal Cloud Run deploy, and to roll back you send traffic back to the previous revision.

we have a dev project and a prod project. dev gets a one percent sample of the real clicks.

monitoring uh is Cloud Monitoring, we have alerts on the Dataflow system lag and on the Pub/Sub oldest unacked message age. they go to the data on-call in Opsgenie.

problems. the the big one is the Dataflow job falling behind after a traffic spike, system lag climbs and the dashboards go stale. usually it catches up by itself, if it doesn't, raise the max workers. and um the other thing is BigQuery streaming insert quota errors on sale days, we switched most tables to the storage write API but two old tables still use streaming inserts.

backups wise, BigQuery tables have seven days of time travel and the raw clicks are also archived to Cloud Storage as Avro files, so we can replay any day. we've, uh, we've never had to replay more than a day though.

security, the collector only accepts requests with a signed token from the app, and personal data, um, like IP addresses, is hashed before it's written anywhere.

whatever you do don't, don't delete the Pub/Sub subscription, the Dataflow job loses its position and you lose clicks.

cost, Dataflow is the biggest line, it's on streaming engine with autoscaling, and we cap max workers at twenty so a spike can't blow the budget.

the busy days are Black Friday and the big summer sale, scale up before them, don't wait for the lag.

owners, it's the data platform team, and the dashboards themselves belong to the growth analytics team. if it's on fire call me until friday and after that call Lena Fischer, she's the data platform lead.

oh, one thing left, those two old tables still need moving to the storage write API, it's on the backlog but nobody's picked it up.

for your first week get the BigQuery and Dataflow roles in both projects and join the Opsgenie schedule. in the first month try an update of the Dataflow job yourself in dev, then in prod with me.""",
    "facts": [
        {"id": "F01", "quote": "it takes every click from the website and the app and gives the product managers near real time dashboards", "in": ["system_overview"]},
        {"id": "F02", "quote": "it's not customer facing but the growth team stares at it all day", "in": ["system_overview"]},
        {"id": "F03", "quote": "the clicks come in through a small Go collector on Cloud Run", "in": AR},
        {"id": "F04", "quote": "it publishes to Pub/Sub", "alt": ["it publishes to Pub Sub"], "in": AR},
        {"id": "F05", "quote": "a Dataflow streaming job does the sessionising and writes into BigQuery", "in": AR},
        {"id": "F06", "quote": "the dashboards are Looker Studio on top of BigQuery", "in": AR + ["monitoring_observability"]},
        {"id": "F07", "quote": "there's a Bigtable table for the live counters", "in": AR},
        {"id": "F08", "quote": "we deploy it with a Cloud Build trigger", "in": ["deployment_and_rollback"]},
        {"id": "F09", "quote": "it does an update in place of the running job", "in": ["deployment_and_rollback"]},
        {"id": "F10", "quote": "you have to drain the old job and start a new one", "in": ["deployment_and_rollback"]},
        {"id": "F11", "quote": "to roll back you send traffic back to the previous revision", "alt": ["to rollback you send traffic back to the previous revision"], "in": ["deployment_and_rollback"]},
        {"id": "F12", "quote": "we have a dev project and a prod project", "in": ["environments"]},
        {"id": "F13", "quote": "dev gets a one percent sample of the real clicks", "in": ["environments"]},
        {"id": "F14", "quote": "we have alerts on the Dataflow system lag and on the Pub/Sub oldest unacked message age", "alt": ["alerts on the Dataflow system lag"], "in": ["monitoring_observability"]},
        {"id": "F15", "quote": "they go to the data on-call in Opsgenie", "in": ["monitoring_observability"]},
        {"id": "F16", "quote": "the Dataflow job falling behind after a traffic spike", "in": ["common_failures"]},
        {"id": "F17", "quote": "system lag climbs and the dashboards go stale", "in": ["common_failures"]},
        {"id": "F18", "quote": "raise the max workers", "in": ["common_failures"]},
        {"id": "F19", "quote": "BigQuery streaming insert quota errors on sale days", "in": ["common_failures"]},
        {"id": "F20", "quote": "two old tables still use streaming inserts", "in": ["common_failures", "open_responsibilities"]},
        {"id": "F21", "quote": "BigQuery tables have seven days of time travel", "in": ["disaster_recovery"]},
        {"id": "F22", "quote": "the raw clicks are also archived to Cloud Storage as Avro files", "in": ["disaster_recovery"]},
        {"id": "F23", "quote": "we've never had to replay more than a day though", "in": ["disaster_recovery"]},
        {"id": "F24", "quote": "the collector only accepts requests with a signed token from the app", "in": ["security_controls"]},
        {"id": "F25", "quote": "is hashed before it's written anywhere", "in": ["security_controls"]},
        {"id": "F26", "quote": "delete the Pub/Sub subscription", "alt": ["delete the Pub Sub subscription"], "in": ["danger_zones"]},
        {"id": "F27", "quote": "Dataflow is the biggest line", "in": ["cost_optimization"]},
        {"id": "F28", "quote": "we cap max workers at twenty so a spike can't blow the budget", "in": ["cost_optimization"]},
        {"id": "F29", "quote": "the busy days are Black Friday and the big summer sale", "in": ["known_bad_days"]},
        {"id": "F30", "quote": "scale up before them, don't wait for the lag", "alt": ["vertical scaling before them, don't wait for the lag"], "in": ["known_bad_days", "common_failures"]},
        {"id": "F31", "quote": "it's the data platform team", "in": ["ownership_escalation"]},
        {"id": "F32", "quote": "the dashboards themselves belong to the growth analytics team", "in": ["ownership_escalation"]},
        {"id": "F33", "quote": "call Lena Fischer, she's the data platform lead", "in": ["ownership_escalation"]},
        {"id": "F34", "quote": "those two old tables still need moving to the storage write API", "in": ["open_responsibilities"]},
        {"id": "F35", "quote": "get the BigQuery and Dataflow roles in both projects and join the Opsgenie schedule", "in": ["day1_survival_checklist", "first_30_day_ownership"]},
        {"id": "F36", "quote": "try an update of the Dataflow job yourself in dev, then in prod with me", "in": ["first_30_day_ownership"]},
    ],
    "covered": ["system_overview", "architecture_reference", "environments", "deployment_and_rollback", "monitoring_observability",
                "common_failures", "disaster_recovery", "security_controls", "danger_zones", "cost_optimization", "known_bad_days",
                "ownership_escalation", "open_responsibilities", "day1_survival_checklist", "first_30_day_ownership"],
    "missing": [],
})


def main(out_dir):
    sys.path.insert(0, r"C:\Users\dell\Continumm\KT_Planner")
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
