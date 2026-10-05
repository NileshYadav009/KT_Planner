#!/usr/bin/env python3
"""
Generate audio from DevOps KT transcript using text-to-speech.

This creates an MP3 audio file that can be uploaded to the KT Planner API.
"""

import pyttsx3

# Your full DevOps KT transcript
transcript = """
Hi everyone, today this KT is about DevOps. 

We will be discussing how we can optimize the KT and how we can make it easy for everybody 
who is giving or who is taking the handover. So there would be no gaps. Like anybody who is 
taking the place can take up a task from day one. 

So let me start with the system overview. The system name is Cloud Native Order Processing Platform. 

This system handles order intake, validation, payment, orchestration and fulfillment triggers. 

It is used by B2C users through the web app, B2B platform partners through APIs and also 
internal teams like finance and support. 

This system is most critical during business hours, specifically during peak sales events. 

The business impact if the system is down is very high, revenue loss, customer dissatisfaction 
and potential compliance issues. 

The worst case failure scenario is orders being accepted but not processed, which leads to 
financial reconciliation problems. 

The problem this system solves is reliable, scalable, order processing across multiple sales channels. 

Every revenue generating flow passes through this platform. 

In terms of business criticality, this system is high. 

If the system is down, order placement breaks, payment confirmation stops, and downstream 
fulfillment is blocked. 

Customers, finance teams, warehouse operations and leaders dashboards are affected. 

We have prod, staging, QA and Dev environments. 

Dev and DevOps own prod and staging while Dev and QA are mostly supported environments.

The key technologies used here include AWS as the cloud provider. All the initiatives for 
continuous integration and deployment. Terraform for infrastructure provisioning. Jenkins 
for CI/CD pipeline. These tools together form the core cloud platform and tooling for this system. 

There is detailed architectural documentation available on Confluence. The document was last 
updated last quarter and has been verified by the incoming owner as part of the KT. 

I strongly recommend reviewing the architecture diagram before touching production. 

Some tribal knowledge. There are some shortcuts in non-prod that do not exist in prod. 

One sharp edge is that manual changes in AWS can drift the Terraform state file. 

Historically, this system had scaling issues during flash sales, so be cautious during traffic spikes. 

Now the DevOps Survival Checklist. The checklist requires access including cloud console access, 
git repository access, CI/CD tool, monitoring, and secrets location. 

The first safe actions you should perform are red zone checks, pipeline view in Jenkins, 
dashboard review in monitoring, and dry running Terraform. 

Then actions not to perform initially include running Terraform apply in prod, editing any 
kinds of secrets, and restarting Kubernetes services blindly. 

Let's move to the normal deployment process. The exact steps are as follows. 

Step one. Code is merged into main branch. 

Step two. Jenkins pipeline triggers automatically. 

Step three. Artifacts are built and deployed in Kubernetes. 

The repo link and pipeline links are documented. 

The trigger type is auto for non-prod and manual for prod. 

The deployment window is outside peak business hours. 

Pre-deployment checks include pipeline green status and no active incidents. 

The deployment validation includes health checks and synthetic monitoring. 

Rollback procedure. If something goes wrong, here is the rollback procedure. 

The rollback trigger is failed health checks or increased error rate. 

The rollback action is reverting the Helm release. 

The risk level is medium if done within 10 minutes. 

The on call lead or engineering managers approve rollback. 

The expected rollback time should be under 15 minutes. 

A common issue in prod is pod crashing or looping due to misconfigured environment variables. 

The likely cause is missing secrets and the fix is validating the secret locations. 

In terms of frequency, this is medium. 

High traffic during peak sales month and EOD audit window are known by design windows. 

During this time, deployment must be avoided. 

As a danger zone, areas you should not touch include production Terraform state files, as it 
can impact the production system and Kubernetes cluster autoscaler settings. 

These are dangerous because they can cause system outages. 

Approvals are required and these actions are emergency only. 

If unsure, do not proceed and escalate. 

The order services are owned by the platform team. 

DevOps should be contacted during deployment or incidents. 

The escalation channel is the on call select channel. 

The first 30 day ownership plan is observe. Week one should be observed, shadow and read only. 

Week two non-prod changes, prod deployment with supervision. Independent ownership in week four. 

Recording open responsibilities and transition plan. Only existing and in progress tasks are 
handed over. No new initiatives. 

The incoming owner can accept, defer or reject tasks, it is up to the upcoming owners. 

If you are comfortable to take up the task, then only say yes, otherwise state formally say no. 

Because this KT will take care of it. It will have proof of what you told in the call. 

So if you have any concerns of any task and if you don't want to take up any task, you are 
free to choose. 

Handover completion. The replacement can deploy safely. The replacement understands rollback. 
The replacement knows danger zones. Escalation paths are clear. Architecture is verified by the owner. 

So this is the small overview which we planned and this is a small KT sample which I will be 
using as my first KT planner continuum application. 

I hope this is sufficient for my application to filter the necessary information. 

I think this is enough. So one last thing, this is a good application. 

Please feel free to reach out to me and test the application as well as possible and let me 
know if you can help me improve it. 

Thank you so much.
"""

def generate_audio(text: str, output_file: str = "devops_kt.mp3"):
    """Generate MP3 audio from text using pyttsx3."""
    print(f"🎤 Generating audio file: {output_file}")
    print("   This may take a minute...")
    
    engine = pyttsx3.init()
    
    # Set speed (default is too fast)
    engine.setProperty('rate', 150)  # 150 words per minute
    
    # Set volume
    engine.setProperty('volume', 0.9)
    
    # Save to file
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    
    print(f"✅ Audio file created: {output_file}")
    print("")
    print("📝 Next steps:")
    print("   1. Run the upload script:")
    print(f"      python upload_and_get_kt.py {output_file} --markdown kt_output.md")
    print("")
    print("   2. The script will:")
    print("      - Upload the audio to the API")
    print("      - Wait for transcription & processing")
    print("      - Download structured KT")
    print("      - Save as Markdown (kt_output.md)")
    print("")

if __name__ == '__main__':
    generate_audio(transcript)
