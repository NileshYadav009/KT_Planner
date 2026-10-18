TRANSCRIPT = """Kiran: This is the handover for Harbor, the shipment tracking service our logistics customers use to follow their parcels.
Harbor is a set of Go services on Amazon ECS behind an Application Load Balancer, and the tracking events land in Amazon Aurora PostgreSQL.
Carrier updates come in through an SQS queue, and a worker service writes them to Aurora.
We deploy with GitHub Actions: a merge to main builds the image and pushes it to ECR, and then the pipeline updates the ECS service.
To roll back, we redeploy the previous task definition from the ECS console.
Grafana shows queue depth and API latency, and an alert fires when the SQS queue is older than ten minutes.
When that alert fires, scale the worker service to six tasks and check the carrier API status page.
The platform team owns the ECS cluster, and my team owns the Go services.
If something breaks at night, page the on-call through PagerDuty, and if they do not answer in fifteen minutes, call Maria, the engineering manager.
Right now we run in one region, us-east-1.
Next quarter we plan to add a second region in eu-west-1, but nobody has started on it.
Aurora snapshots are taken every night and kept for seven days.
We have never tested a restore.
Never run the database migration script against production without a snapshot first, it drops the old columns.
The Black Friday week is the busiest time, so there are no deployments in that week.
Customer data is encrypted at rest with KMS, and only the support team can see parcel recipient names.
The biggest cost is Aurora, about three thousand dollars a month.
The API rate limit is 500 requests a minute per customer.
Sorry, not 500, the limit is 300 requests a minute per customer.
On your first day, get access to the AWS account and the PagerDuty rota.
In your first month, you should take over the weekly release.
"""
