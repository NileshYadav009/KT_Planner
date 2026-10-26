"""Tests for devops_transcription.py's fuzzy term-correction step.

Regression guard for a real data-corruption bug found via a 4-transcript
adversarial test pass: apply_fuzzy_term_corrections() tokenized on
`\\b[\\w/]+\\b`, which doesn't include apostrophes, so any contraction
("it's", "that's", "there's") split into two tokens ("it" + "s"). The
resulting bare "s" token then formed n-grams like "s the" that fuzzy-
matched the known glossary term "s three" (used to correct mis-heard
"S3") with a very high jaro-winkler score, silently corrupting ordinary
sentences like "It's the old claims processing system" into "It's three
old claims processing system". This wasn't a rare edge case — "it's
the"/"that's the"/"there's a" are among the most common contraction
patterns in spoken English, so this could corrupt real transcript content
on a routine basis.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from devops_transcription import clean_transcript, apply_devops_corrections


def test_apostrophe_contraction_not_corrupted_by_s3_fuzzy_match():
    text = "It's the old claims processing system, it's been around forever."
    result = clean_transcript(text)
    assert "three" not in result.lower()
    assert "It's the old claims processing system" in result


def test_various_common_contractions_survive_fuzzy_correction():
    for contraction_sentence in [
        "It's the primary database for the platform.",
        "That's the escalation path we use.",
        "There's the danger zone everyone talks about.",
        "What's the rollback procedure look like?",
    ]:
        corrected, _ = apply_devops_corrections(contraction_sentence)
        assert "three" not in corrected.lower(), f"corrupted: {contraction_sentence!r} -> {corrected!r}"


def test_genuine_s3_mention_without_apostrophe_is_unaffected():
    # This fix only removes the spurious apostrophe-splitting trigger — a
    # literal "s three" (no apostrophe involved at all) tokenizes and is
    # evaluated exactly as before.
    text = "We store all our backups in s three for durability."
    corrected, _ = apply_devops_corrections(text)
    assert "s three" in corrected.lower() or "s3" in corrected.lower()


def test_providers_not_corrupted_into_process_by_prefix_bias():
    # Found via a real transcript (AWS e-commerce KT): jaro-winkler weights a
    # shared prefix heavily, so "providers" scores 0.83 against the known
    # glossary term "process" (both start "pro...") even though the words
    # mean completely different things -- silently corrupting "Some payment
    # providers are mocked in staging" into "Some payment process are mocked
    # in staging". A plain Levenshtein similarity check (0.56 for this pair,
    # well below genuine corrections like "rabbitmq"/"rabbit" at 0.75) isn't
    # fooled by the shared prefix and blocks it.
    from devops_transcription import apply_fuzzy_term_corrections
    text = "Some payment providers are mocked in staging."
    corrected, corrections = apply_fuzzy_term_corrections(text)
    assert "provider" in corrected.lower()
    assert "process" not in corrected.lower()
    assert corrections == []


def test_ordinary_phrase_close_to_a_term_word_by_word_is_left_alone():
    # "log hours" passes the per-word checks against "long horn" but is a
    # different phrase; it was rewritten to "longhorn" (a storage product).
    from devops_transcription import apply_fuzzy_term_corrections
    text = "Around four hundred consultants log hours in it every week."
    corrected, corrections = apply_fuzzy_term_corrections(text)
    assert corrected == text
    assert corrections == []


def test_a_possessive_is_not_a_garbled_term():
    # "Stripe's test" scored 0.92 against "stress test" and passed every
    # per-word check: the payment provider's test mode became a stress test.
    from devops_transcription import apply_fuzzy_term_corrections
    text = "Dev uses Stripe's test mode, so no real money moves."
    corrected, corrections = apply_fuzzy_term_corrections(text)
    assert corrected == text
    assert corrections == []


def test_genuine_multiword_fuzzy_corrections_still_work():
    # Guards against the Levenshtein guard being too strict and blocking
    # real corrections it wasn't meant to touch. Now lands on the canonical
    # key "rabbitmq" (not just the intermediate registered variant "rabbit
    # mq") since get_canonical_key_map() correction now normalizes a
    # matched variant to its canonical spelling — see
    # test_known_variant_is_normalized_to_canonical_key_not_left_alone
    # below for the bug this specifically fixes.
    from devops_transcription import apply_fuzzy_term_corrections
    text = "We rely on rabid mq heavily for async messaging."
    corrected, corrections = apply_fuzzy_term_corrections(text)
    assert "rabbitmq" in corrected.lower()
    assert corrections and corrections[0]["corrected"] == "rabbitmq"


def test_known_variant_is_normalized_to_canonical_key_not_left_alone():
    # Regression test for a real bug: DEVOPS_VOCABULARY's variant lists are
    # documented as "the ways Whisper is likely to mis-transcribe" the
    # canonical term, but a listed variant used to be treated as "already a
    # known term, nothing to correct" — apply_fuzzy_term_corrections left
    # "pager duty" and "rabbit mq" completely untouched even though both are
    # registered variants of "pagerduty"/"rabbitmq", not the canonical forms
    # themselves.
    from devops_transcription import apply_fuzzy_term_corrections
    text = "pager duty is used for alerting and rabbit mq handles messaging."
    corrected, corrections = apply_fuzzy_term_corrections(text)
    assert "pagerduty" in corrected.lower()
    assert "pager duty" not in corrected.lower()
    assert "rabbitmq" in corrected.lower()
    assert "rabbit mq" not in corrected.lower()


def test_already_canonical_terms_are_left_alone():
    # The canonical spelling itself must never be "corrected" into anything
    # else — only listed variants should be normalized.
    from devops_transcription import apply_fuzzy_term_corrections
    text = "PagerDuty is used for alerting and rabbitmq handles messaging."
    corrected, corrections = apply_fuzzy_term_corrections(text)
    assert corrected == text
    assert corrections == []


def test_pub_slash_sub_corrected_to_product_name():
    # Regression test: GCP Pub/Sub is routinely spelled out verbally ("pub
    # slash sub") since "/" can't be spoken — a real generated KT left this
    # uncorrected throughout an entire document (table rows, checklist
    # items, danger zones all read "pub slash sub" verbatim).
    text, _ = apply_devops_corrections("Review pub slash sub before making changes.")
    assert "Pub/Sub" in text
    assert "pub slash sub" not in text.lower()


def test_graphana_typo_corrected_to_grafana():
    # A live generated KT showed "Graphana dashboards" throughout, even
    # though "Grafana" (correct spelling) appeared correctly elsewhere in
    # the same document — the fuzzy multi-word corrector deliberately
    # doesn't touch single words (see the "scanning"->"scaling"
    # corruption this session reverted), so this needs an exact,
    # single-token PHRASE_CORRECTIONS entry instead, same class of fix as
    # "trevi" -> "Trivy".
    text, _ = apply_devops_corrections("Review Graphana dashboards for the incident.")
    assert "Grafana" in text
    assert "Graphana" not in text


def test_pager_duty_spacing_corrected_to_product_name():
    text, _ = apply_devops_corrections("pager duty is used for alerting.")
    assert "PagerDuty" in text


def test_argo_cd_with_space_corrected_to_product_name():
    # "Argo CD" (its own official stylization, two words) is at least as
    # common in real speech as "ArgoCD" — must normalize the same way.
    text, _ = apply_devops_corrections("Argo CD handles GitOps deployment.")
    assert "ArgoCD" in text


def test_brand_name_casing_batch_normalized_to_proper_product_names():
    # A curated batch covering the highest-traffic devops_vocabulary.py
    # entries whose spoken form commonly splits into separate words —
    # these should reach real branded capitalization (via PHRASE_CORRECTIONS,
    # which runs before the fuzzy safety net), not just a lowercase,
    # word-glued canonical key.
    cases = {
        "We use mongo db for storage.": "MongoDB",
        "Data lands in dynamo db.": "DynamoDB",
        "Deployment uses code pipeline and code build.": ("CodePipeline", "CodeBuild"),
        "Monitoring uses open telemetry and sonar qube.": ("OpenTelemetry", "SonarQube"),
        "Alerts route through ops genie.": "OpsGenie",
        "We use rabbit mq and active mq together.": ("RabbitMQ", "ActiveMQ"),
        "Azure cosmos db is our primary store.": "Azure Cosmos DB",
        "Rollback is handled by octopus deploy.": "Octopus Deploy",
    }
    for text, expected in cases.items():
        corrected, _ = apply_devops_corrections(text)
        expected_terms = (expected,) if isinstance(expected, str) else expected
        for term in expected_terms:
            assert term in corrected, f"expected {term!r} in {corrected!r} (from {text!r})"


def test_trivi_typo_corrected_to_trivy():
    # "Trivi" (one-vowel swap from "Trivy") slipped through uncorrected in a
    # live AWS KT's Security section ("Trivi for security scanning"), even
    # though the sibling mishearing "trevi" was already handled.
    text, _ = apply_devops_corrections("Security scanning uses Trivi for container images.")
    assert "Trivy" in text
    assert "Trivi" not in text


def test_certificate_xberry_corrected_to_certificate_expiry():
    # "Certificate XBerry" is Whisper mishearing "certificate expiry" --
    # confirmed in two live Azure Banking KTs, where a Common Failures row
    # rendered as the literal nonsense phrase "Certificate XBerry" with
    # every other cell blank (no likely cause/fix could match a symptom
    # name that isn't a real phrase).
    text, _ = apply_devops_corrections("Another common issue is Certificate XBerry.")
    assert "certificate expiry" in text.lower()
    assert "xberry" not in text.lower()


def test_rural_metadata_corrected_to_raw_metadata():
    # "Rural metadata" is Whisper mishearing "raw metadata" -- confirmed in
    # a live GCP KT's Disaster Recovery section ("Rural metadata backed up
    # daily"), a nonsensical phrase in a devops context.
    text, _ = apply_devops_corrections("Rural metadata backed up daily.")
    assert "raw metadata" in text.lower()
    assert "rural" not in text.lower()


def test_drive_testing_corrected_to_dr_testing():
    # "drive testing" is Whisper mishearing "DR testing" (disaster-recovery
    # testing) -- disaster_recovery's own schema hints list "dr testing" as
    # the expected phrase, so the mishearing fails to match anything.
    text, _ = apply_devops_corrections("Drive testing is performed twice a year.")
    assert "dr testing" in text.lower()
    assert "drive testing" not in text.lower()


def test_bicep_casing_corrected_to_branded_name():
    text, _ = apply_devops_corrections("We manage infrastructure with bicep templates.")
    assert "Bicep" in text
    assert "bicep templates" not in text


def test_crash_loop_back_off_corrected_to_crashloopbackoff():
    text, _ = apply_devops_corrections("Pods sometimes enter a crash loop back off state.")
    assert "CrashLoopBackOff" in text
    assert "crash loop back off" not in text.lower()


def test_azure_cash_for_redis_corrected_to_cache():
    # "Cash" is Whisper mishearing "Cache" -- confirmed live in two Azure
    # KTs, where the nonsense phrase reached the rendered document.
    text, _ = apply_devops_corrections("Azure Cash for Redis provides caching.")
    assert "Azure Cache for Redis" in text
    assert "cash" not in text.lower()


def test_doubled_azure_and_lowercase_front_door_corrected():
    # "Azure azure front door" is a real transcript artifact. Both the
    # doubled word and the brand casing must be fixed deterministically --
    # this used to correct only in sections the LLM polish pass touched,
    # leaving Additional Notes (raw text) showing the broken form.
    text, _ = apply_devops_corrections(
        "Traffic flows through Azure azure front door and Application Gateway to AKS."
    )
    assert "Azure Front Door" in text
    assert "azure azure" not in text.lower()


def test_azure_service_names_get_branded_casing():
    text, _ = apply_devops_corrections(
        "azure sql is the primary database and azure service bus handles messages, "
        "secrets live in azure key vault and azure devops runs ci."
    )
    for expected in ("Azure SQL", "Azure Service Bus", "Azure Key Vault", "Azure DevOps"):
        assert expected in text, expected


def test_brand_casing_corrections_do_not_misfire_on_ordinary_english_phrases():
    # Terms deliberately excluded from the batch above because their
    # "variant" spelling is also an ordinary English word/phrase (e.g.
    # devops_vocabulary.py lists "customize" as a mishearing of "kustomize")
    # — must never get swept into a devops-specific correction.
    text = "Please customize the report before the team meeting."
    corrected, _ = apply_devops_corrections(text)
    assert "customize" in corrected.lower()
    assert "kustomize" not in corrected.lower()


def test_two_word_product_names_are_joined_only_where_they_are_the_product():
    from devops_transcription import clean_transcript

    aws = clean_transcript("Logs go to Amazon open search, and the open search domain has three nodes.")
    assert aws.count("OpenSearch") == 2
    assert "open search results" in clean_transcript("Please open search results in a new tab.")
    gcp = clean_transcript("Events land in Pub/Sub, and the big query tables are partitioned daily.")
    assert "BigQuery tables" in gcp
    assert "a big query" in clean_transcript("Running a big query on the reporting database is slow.")


def test_argo_city_is_argocd_only_where_it_deploys():
    """'Argo City' is a common mishearing of ArgoCD; corrected only where the
    sentence is about deploying, so the intended term is unambiguous."""
    from devops_transcription import clean_transcript

    assert "ArgoCD deploys them" in clean_transcript("GitHub Actions builds the images and Argo City deploys them.")
    assert "deployed through ArgoCD" in clean_transcript("Everything is deployed through Argo City.")
    assert "Argo City" in clean_transcript("Our biggest customer is in Argo City.")


# --------------------------------------------------------------------------
# A recorded KT spoken with deliberate mistakes (14 Oct): misheard tool
# names stayed in the document, one "sentence" ran 90 words, and the tour
# narration and closing remarks were kept.
# --------------------------------------------------------------------------

def test_sound_alikes_of_devops_terms_are_corrected():
    from devops_transcription import clean_transcript

    out = clean_transcript(
        "The services uses post gray SQL on Amazon RDS for order data and ready for caching GitHub Action Built "
        "builds the application and Argo UD deploys them to eCase. PromoTales and Grafana monitor application "
        "health. Never modify data form code manually. Then we see the nodes, the ports running on them. "
        "How many ports are there on each node? The HPA, Main Max, the CPU target. The namespecies are here.")
    for expected in ("PostgreSQL on Amazon RDS", "Redis for caching", "GitHub Actions builds the application",
                     "ArgoCD deploys them to EKS", "Prometheus and Grafana", "Never modify Terraform code",
                     "the pods running on them", "How many pods are there on each node", "min/max", "namespaces"):
        assert expected in out, expected


def test_the_ordinary_words_stay_where_they_are_ordinary():
    from devops_transcription import clean_transcript

    out = clean_transcript("Open port 443 on the load balancer. The data form on the website is ready for review. "
                           "We use GitHub Actions and Argo CD. Postgres is fine.")
    for kept in ("port 443", "The data form on the website is ready for review", "GitHub Actions", "Postgres is fine"):
        assert kept in out, kept


def test_a_run_on_sentence_is_split_at_its_sentence_openers():
    from context_mapper import AudioSegment, segment_sentences, split_run_on

    run_on = ("I am handing over our online grocery platform It processes around 15,000 orders daily customers use a "
              "react application with requests going through CloudFront and an application load balancer to services "
              "running on Amazon EKS The services use PostgreSQL on Amazon RDS for order data.")
    assert split_run_on(run_on)[:2] == ["I am handing over our online grocery platform",
                                        "It processes around 15,000 orders daily customers use a react application with "
                                        "requests going through CloudFront and an application load balancer to services "
                                        "running on Amazon EKS"]
    texts = [s.text for s in segment_sentences([AudioSegment(text=run_on, start=0, end=9, avg_logprob=-0.3)])]
    assert "I am handing over our online grocery platform." in texts
    # Not merged back: the rebuilt run-on went whole to Deployment, business
    # volume and all.
    assert "The services use PostgreSQL on Amazon RDS for order data." in texts
    # A sentence of normal length is left alone.
    short = "Airflow handles orchestration and The scheduler retries tasks."
    assert split_run_on(short) == [short]


def test_a_segment_that_ends_mid_sentence_gets_no_full_stop():
    # A recording is cleaned one Whisper segment at a time: "stores container
    # images and. ArgoCD deploys them" came from a full stop added to each.
    from devops_transcription import clean_transcript

    assert clean_transcript("Amazon ECR stores container images and") == "Amazon ECR stores container images and"


def test_closing_remarks_at_the_end_are_trimmed():
    from devops_transcription import clean_transcript

    assert clean_transcript("There are three recommendations by category so I hope it is very clear it is very useful "
                            "for you so please if you have any doubts any questions let me know thank you") == \
        "There are three recommendations by category."
    assert clean_transcript("Backups run nightly. Thank you.") == "Backups run nightly."
    # A recording's last segment: only a stub is left, and it is dropped.
    assert clean_transcript("For you so please if you have any doubts any questions let me know thank you") == ""
    assert clean_transcript("Airflow handles orchestration. Thanks to Redis the cache is warm.") == \
        "Airflow handles orchestration. Thanks to Redis the cache is warm."


def test_spoken_agreement_and_turn_fillers_are_removed_without_commas():
    from devops_transcription import clean_transcript

    assert clean_transcript("Then we have the request throughput which reached 15,000 right so yep so we have "
                            "three active alerts") == \
        "Then we have the request throughput which reached 15,000. We have three active alerts"
    assert clean_transcript("Yeah, the cluster runs on EKS.") == "The cluster runs on EKS."
    assert clean_transcript("The latency is fine okay so now we check alerts") == \
        "The latency is fine. Now we check alerts"
    # "right" as an answer or a direction is kept.
    assert clean_transcript("That is right so it works") == "That is right so it works"
    assert clean_transcript("Turn right so the traffic flows") == "Turn right so the traffic flows"


def test_a_transcribed_piece_is_closed_when_the_next_one_opens_a_sentence():
    from devops_transcription import close_segment_sentences

    assert close_segment_sentences(["everything in dev is moved on to production", "Then we can see the health",
                                    "and we have", "The rollout history."]) == \
        ["everything in dev is moved on to production.", "Then we can see the health",
         "and we have", "The rollout history."]


def test_correct_wording_is_left_as_said_and_mishearings_are_still_fixed():
    """The vocabulary listed real phrases as if they were mishearings, so
    "scale out the web tier" became "horizontal scaling the web tier", "The
    recovery time objective" became "The rto" and "Entra ID" became the
    retired "azure active directory" (audit 9aak)."""
    from devops_transcription import clean_transcript

    for text, phrase in [
        ("We scale out the web tier before the sale.", "scale out"),
        ("Scale up the database before the sale.", "scale up the database"),
        ("We scaled out the web tier last year.", "scaled out"),
        ("The recovery time objective is four hours.", "recovery time objective"),
        ("We use a feature toggle and a canary release.", "feature toggle and a canary release"),
        ("Staff sign in with Entra ID.", "entra id"),
        ("The certificate manager renews the certs.", "certificate manager"),
        ("Alert routing sends pages to the on-call phone.", "alert routing"),
        ("Pods run on Azure Kubernetes Service and Google Cloud.", "azure kubernetes service and google cloud"),
    ]:
        assert phrase in clean_transcript(text).lower(), text
    assert "Kafka" in clean_transcript("coffee for a handles the order events.")
    assert "Prometheus" in clean_transcript("promo theus scrapes the metrics.")
    assert "neo4j" in clean_transcript("The neo four j cluster.").lower()


def test_actually_is_kept_where_it_corrects_what_was_just_said():
    """Opening a sentence it marks a correction; inside one it is emphasis."""
    from devops_transcription import clean_transcript

    assert clean_transcript("The RTO is four hours. Actually the RTO is thirty minutes.") == \
        "The RTO is four hours. Actually the RTO is thirty minutes."
    assert clean_transcript("It was actually down for an hour.") == "It was down for an hour."
    assert clean_transcript("Um, okay, so, uh, this is the KT for Pulse.") == "This is the KT for Pulse."
    # Before a number it corrects one, and stays.
    assert clean_transcript("The RTO is four hours, actually two hours.") == \
        "The RTO is four hours, actually two hours."


def test_spelled_out_acronyms_and_spaced_tool_names_are_joined():
    """Adversarial matrix L02: speech-to-text wrote "r d s", "post gres" and
    "graph ana", and the document kept them."""
    from devops_transcription import clean_transcript

    assert clean_transcript("it runs on e k s and messages go to s q s") == "It runs on EKS and messages go to SQS"
    assert clean_transcript("the post gres database runs on r d s") == "The Postgres database runs on RDS"
    assert clean_transcript("we use post gres sql, and the dashboards are in graph ana") == \
        "We use PostgreSQL, and the dashboards are in Grafana"
    # Letters that are not a known acronym, and ordinary single letters.
    for said in ["We ran an a b test on the checkout page.", "I am the owner of the queue.", "Plan b is a rollback."]:
        assert clean_transcript(said) == said, said


def test_a_name_opening_a_sentence_keeps_its_spelling():
    """Adversarial matrix L01: "orders-api handles checkout requests" became
    "Orders-api ...", and "iOS" became "IOS"."""
    from devops_transcription import clean_transcript

    said = ("This is the handover. orders-api handles checkout requests. kubectl rollout restart fixes it. "
            "iOS clients call it. payments_worker drains the queue. on-call engineers get paged. the queue is SQS.")
    assert clean_transcript(said) == (
        "This is the handover. orders-api handles checkout requests. kubectl rollout restart fixes it. "
        "iOS clients call it. payments_worker drains the queue. On-call engineers get paged. The queue is SQS.")


def test_a_number_corrected_mid_sentence_keeps_only_the_corrected_one():
    from devops_transcription import resolve_self_corrections

    corrected = {
        "That the order limit is 4000 or sorry not 4000 but 2000.": "That the order limit is 2000.",
        "The order limit is 4,000, sorry, 2,000 per day.": "The order limit is 2,000 per day.",
        "The RTO is four hours, I mean two hours.": "The RTO is two hours.",
        # The unit was said only with the withdrawn number.
        "The RTO is four hours, I mean two.": "The RTO is two hours.",
        "We keep backups for 30 days, no wait, 14 days.": "We keep backups for 14 days.",
        "Each pod gets 2GB or rather 4GB of memory.": "Each pod gets 4GB of memory.",
        "Availability target is 99.9%, sorry, 99.95%.": "Availability target is 99.95%.",
        "There are three hundred and fifty, sorry, four hundred tenants.": "There are four hundred tenants.",
        "The limit is 4000, sorry I mean it's 2000.": "The limit is 2000.",
        "Retries happen 5 times, correction, 3 times.": "Retries happen 3 times.",
    }
    for said, written in corrected.items():
        assert resolve_self_corrections(said) == written, said
    # No number corrected by a number: as said.
    for said in ["We had 500 errors, sorry for the noise.",
                 "We prefer 200 rather than 300 connections.",
                 "It takes 5 minutes, actually 10 minutes on a bad day.",
                 "Weekdays 9 to 5, no 24/7 support.",
                 "Sorry, I should say the price lists come in from SAP.",
                 "The old one, sorry, the new one is in us-east-1.",
                 "Sorry, not 4000 but 2000."]:
        assert resolve_self_corrections(said) == said, said


def test_a_correction_after_a_dash_or_restating_its_lead_in_keeps_only_the_corrected_value():
    """Adversarial matrix A01, A04: a dash before the marker, "no, sorry",
    and the words before the number said again ("every 30 minutes, sorry,
    every 30 seconds") left both values in the document."""
    from devops_transcription import resolve_self_corrections

    corrected = {
        "The cache limit is 2000K—no, sorry, it's 3000K.": "The cache limit is 3000K.",
        "Restart the service every 30 minutes—sorry, every 30 seconds—until it reports healthy.":
            "Restart the service every 30 seconds—until it reports healthy.",
        "The limit is 4000 -- sorry, 2000.": "The limit is 2000.",
        "We keep backups for 30 days, sorry, for 14 days.": "We keep backups for 14 days.",
        "The limit is 4000, sorry, the limit is 2000.": "The limit is 2000.",
        "The timeout is 30 seconds, no, sorry, 45 seconds.": "The timeout is 45 seconds.",
    }
    for said, written in corrected.items():
        assert resolve_self_corrections(said) == written, said
    # The words after the marker are not the ones before the number, or no
    # number follows: as said.
    for said in ["Restart every 30 minutes, sorry, every pod restarts in 30 seconds.",
                 "We keep backups for 30 days, sorry, within 14 days of a release.",
                 "The window is 2 hours—sorry for the delay—and it starts at 9.",
                 "Keep 3 replicas, no, sorry, I meant the staging cluster."]:
        assert resolve_self_corrections(said) == said, said


def test_a_correction_in_the_next_sentence_replaces_the_number_it_corrects():
    from devops_transcription import resolve_correction_across as fix

    previous = "The API rate limit is 500 requests a minute per customer."
    assert fix(previous, "Sorry, not 500, the limit is 300 requests a minute per customer.") == \
        "The API rate limit is 300 requests a minute per customer."
    assert fix(previous, "Sorry, 300.") == "The API rate limit is 300 requests a minute per customer."
    assert fix(previous, "I mean 300 requests a minute.") == "The API rate limit is 300 requests a minute per customer."
    assert fix("The nightly job runs at 2 am.", "Actually, 3 am.") == "The nightly job runs at 3 am."
    assert fix("We keep backups for thirty days.", "Sorry, not thirty, fourteen days.") == \
        "We keep backups for fourteen days."
    # Not a correction of a number the sentence before stated, or it says
    # more than the sentence before: both stay as said.
    for nxt in ["Sorry, not 500, it's 300, and partners get 1000.",
                "Sorry, not 400, the limit is 300.",
                "Sorry, I lost my train of thought.",
                "Actually, the rollback takes 10 minutes.",
                "The burst limit is 300."]:
        assert fix(previous, nxt) is None, nxt
    # Two numbers before it, and the correction does not say which.
    assert fix("Staging has 2 nodes and production has 6 nodes.", "Sorry, 8.") is None


def test_a_sentence_that_replaces_a_value_of_the_one_before_is_recognised():
    """Adversarial matrix A02, A03, A05, A09: the next sentence replaces the
    value but says more than the number, so it cannot be rewritten into the
    sentence before; shown apart, the old value read as current."""
    from devops_transcription import replaces_value_before as replaces

    assert replaces("We use PostgreSQL 14.",
                    "Actually, the upgrade is complete; production is now on PostgreSQL 16.")
    assert replaces("The service runs on port 8080.", "Correction, the actual production port is 8443.")
    assert replaces("The database backup retention is 7 days.",
                    "Wait, that's old information; we've changed it to 30 days.")
    assert replaces("The upload size limit is 2000K.", "The upload size limit is 3000K; use the latter.")
    # No correction signal, an apology, "actually" adding a fact, a value of
    # another kind, or a plain number correction (resolve_correction_across).
    for previous, nxt in [
            ("Staging has 2 nodes.", "Production has 6 nodes."),
            ("Retention is 7 days.", "Sorry for the noise, the logs are kept 30 days."),
            ("The deploy takes 5 minutes.", "Actually, the rollback takes 10 minutes."),
            ("We use PostgreSQL 14.", "Sorry, the cache is Redis 7."),
            ("Backups are kept for 7 days.", "Sorry, I lost my train of thought."),
            ("The API rate limit is 500 requests a minute per customer.",
             "Sorry, not 500, the limit is 300 requests a minute per customer.")]:
        assert not replaces(previous, nxt), nxt


def test_a_change_stated_as_news_or_with_a_hedge_replaces_the_value_before():
    """Adversarial matrix A16, A18: "The database is PostgreSQL 14. We
    upgraded production to PostgreSQL 16 last night." and "The backup
    retention is seven days. I think we changed it to thirty days, but I'm
    not sure." have no correction marker; the old value stood alone."""
    from devops_transcription import replaces_value_before as replaces

    assert replaces("The database is PostgreSQL 14.", "We upgraded production to PostgreSQL 16 last night.")
    assert replaces("The backup retention is seven days.", "I think we changed it to thirty days, but I'm not sure.")
    assert replaces("The pool size is 20 connections.", "It is now 40 connections.")
    # A change to something of another kind is a new fact.
    assert not replaces("The pool size is 20 connections.", "We increased the timeout to 30 seconds.")


def test_a_correction_of_a_statement_without_a_number_is_recognised():
    """Adversarial matrix A19: "The service is owned by Platform Engineering.
    Actually, application ownership belongs to Payments, but Platform owns
    the cluster." The first stood alone as the service's owner."""
    from devops_transcription import corrects_statement_before as corrects

    assert corrects("The service is owned by Platform Engineering.",
                    "Actually, application ownership belongs to Payments, but Platform owns the cluster.")
    assert corrects("The payments team owns the service.", "Actually the platform team owns it since the reorg.")
    assert corrects("Alerts go to the ops channel.", "Sorry, alerts go to the payments-oncall channel.")
    # A new topic after "actually", nothing in common, or no marker at all.
    for previous, nxt in [
            ("Monitoring, we have Airflow's own task alerts going to a Slack channel.",
             "Actually that reminds me, the most annoying failure is an empty file from a store."),
            ("The service is owned by Platform Engineering.", "Actually, deploys happen on Tuesdays."),
            ("The service is owned by Platform Engineering.", "Platform also owns the cluster.")]:
        assert not corrects(previous, nxt), nxt


def test_one_value_compared_across_environments_stays_one_statement():
    """Adversarial matrix A17: "Staging uses 2000K, while production uses
    3000K." was cut in two, and "Production uses 3000K." landed in System
    Overview without what it measured."""
    from context_mapper import split_facts

    for said in ["Staging uses 2000K, while production uses 3000K.",
                 "Staging has 2 nodes, and production has 6 nodes."]:
        assert split_facts(said) == [(0, len(said.rstrip()))], said
    # Two settings, or no environments: cut as before.
    for said in ["The rto is four hours, and the rpo is one hour.",
                 "The frontend runs on Vercel, while the API runs on Amazon ECS."]:
        assert len(split_facts(said)) == 2, said


def test_a_replaced_value_is_kept_whole_with_the_sentence_that_replaces_it():
    from context_mapper import AudioSegment, segment_sentences

    said = ("The billing API runs on Amazon ECS. We use PostgreSQL 14. Actually, the upgrade is complete; "
            "production is now on PostgreSQL 16. Deploys go through Jenkins.")
    units = [s.text for s in segment_sentences([AudioSegment(text=said, start=0.0, end=9.0, avg_logprob=-0.2)])]
    assert "We use PostgreSQL 14. Actually, the upgrade is complete; production is now on PostgreSQL 16." in units
    assert not any(u.startswith("We use PostgreSQL 14") and "16" not in u for u in units)
    assert "The billing API runs on Amazon ECS." in units and "Deploys go through Jenkins." in units


def test_a_value_corrected_much_later_is_shown_with_its_correction():
    """Adversarial matrix A11: "I was wrong about the backups earlier" seven
    sentences on; the 14 days stood alone in Disaster Recovery."""
    from context_mapper import AudioSegment, segment_sentences

    said = ("Backups of the staging tables are kept for 14 days. The finance team reads Looker every morning. "
            "Snowflake costs about four thousand dollars a month. If the nightly load fails, rerun the task. "
            "Oh, and I was wrong about the backups earlier, they are kept for 35 days now. Deploys run on Fridays.")
    units = [s.text for s in segment_sentences([AudioSegment(text=said, start=0.0, end=9.0, avg_logprob=-0.2)])]
    assert ("Backups of the staging tables are kept for 14 days. Oh, and I was wrong about the backups earlier, "
            "they are kept for 35 days now.") in units
    assert units.index("The finance team reads Looker every morning.") == 0
    assert units[-1] == "Deploys run on Fridays."
    # Not about what the earlier sentence measured, or no pointer back.
    for later in ["Oh, and I was wrong about the dashboards earlier, they refresh every 35 days now.",
                  "Logs are kept for 35 days now."]:
        units = [s.text for s in segment_sentences([AudioSegment(
            text=f"Backups are kept for 14 days. Snowflake runs the warehouse. {later}",
            start=0.0, end=9.0, avg_logprob=-0.2)])]
        assert units[0] == "Backups are kept for 14 days.", later


def test_the_corrected_sentence_is_merged_and_the_correction_is_recorded():
    from context_mapper import Sentence, merge_spoken_corrections

    said = [Sentence(text="The API rate limit is 500 requests a minute per customer.", start=1.0, end=2.0, speaker="Kiran"),
            Sentence(text="Sorry, not 500, the limit is 300 requests a minute per customer.", start=2.0, end=3.0,
                     speaker="Kiran"),
            Sentence(text="Deployments go through GitHub Actions.", start=3.0, end=4.0, speaker="Kiran")]
    excluded = []
    units, corrected = merge_spoken_corrections(said, excluded)
    assert len(units) == 2
    assert corrected == {0: "The API rate limit is 300 requests a minute per customer."}
    # The unit keeps both sentences as said, for topics and sources.
    assert "500" in units[0].text and "300" in units[0].raw_text and units[0].end == 3.0
    assert excluded and "Sorry, not 500" in excluded[0]["text"]
    # Another speaker's "sorry" is not the speaker correcting themselves.
    other = [said[0], Sentence(text="Sorry, 300.", start=2.0, end=3.0, speaker="Nadia")]
    assert len(merge_spoken_corrections(other, [])[0]) == 2


def test_a_value_withdrawn_later_in_the_same_sentence_is_found():
    from devops_transcription import withdrawn_values

    def pairs(text):
        return [(p["old"], p["new"]) for p in withdrawn_values(text)]

    # A user's KT (23 Oct).
    assert pairs("The database backups are retained for seven days—actually, the new retention policy is "
                 "30 days.") == [("seven", "30")]
    assert pairs("It runs on AWS EKS, and the production database is PostgreSQL 14—sorry, we upgraded it last "
                 "week, so it's PostgreSQL 16 now.") == [("14", "16")]
    assert pairs("The cache limit is 2000K—no, sorry, it's 3000K.") == [("2000k", "3000k")]
    # "actually" adding to a value, not withdrawing it.
    assert pairs("A restore takes 5 minutes, actually 10 on a bad day.") == []
    assert pairs("We keep 3 replicas, sorry for the noise, and 2 workers.") == []


def test_a_correction_is_not_cut_away_from_the_value_it_withdraws():
    # A user's KT (23 Oct): cut after "actually", "The database backups are
    # retained for seven days" became a fact of its own and was listed as a
    # recovery action; the new retention period went into another item.
    from context_mapper import split_facts

    said = "The database backups are retained for seven days—actually, the new retention policy is 30 days."
    assert split_facts(said) == [(0, len(said.rstrip()))]
    # Cut where each part keeps its own correction whole.
    both = ("It runs on AWS EKS, and the production database is PostgreSQL 14—sorry, we upgraded it last week, "
            "so it's PostgreSQL 16 now.")
    pieces = [both[a:b] for a, b in split_facts(both)]
    assert len(pieces) == 2 and "PostgreSQL 14" in pieces[1] and "PostgreSQL 16" in pieces[1]


def test_roll_back_and_fail_over_said_as_verbs_stay_verbs():
    # The user's KTs (24 Oct): "we roll back using the previous ECS task
    # definition" was printed "we rollback using ...".
    for said in ["If a release fails, we roll back using the previous ECS task definition.",
                 "You can roll back to the previous revision.", "We fail over to the secondary region."]:
        assert clean_transcript(said) == said
    assert clean_transcript("Do a roll back if the canary fails.") == "Do a rollback if the canary fails."
    assert clean_transcript("A fail over takes five minutes.") == "A failover takes five minutes."


@pytest.mark.parametrize("heard,meant", [
    # Whisper on the user's recordings (24 Oct).
    ("If a pod enters CrashLuke backoff, verify the secret references.",
     "If a pod enters CrashLoopBackOff, verify the secret references."),
    ("Diplomats must be avoided during those periods.", "Deployments must be avoided during those periods."),
    ("It handles validation, pavement orchestration and fulfillment.",
     "It handles validation, payment orchestration and fulfillment."),
    ("Traffic enters through application roadbalances.", "Traffic enters through application load balancers."),
    ("Jenkins for CSD pipeline.", "Jenkins for CI/CD pipeline."),
    ("Verify secret references before investigating application cold.",
     "Verify secret references before investigating application code."),
    ("Busy during Black Friday, Month and Sales and campaigns.", "Busy during Black Friday, month-end sales and campaigns."),
    ("Check the cluster autoscalar configuration.", "Check the cluster autoscaler configuration."),
    # Said as meant.
    ("The diplomats met in Geneva.", "The diplomats met in Geneva."),
    ("The pavement outside the office is wet.", "The pavement outside the office is wet."),
    ("The application cold start takes two seconds.", "The application cold start takes two seconds."),
])
def test_mishearings_on_real_recordings_are_corrected_in_context(heard, meant):
    assert clean_transcript(heard) == meant


def test_a_tag_question_asks_in_a_conversation_and_is_a_statement_in_a_talk():
    # In a conversation "..., right?" asks the giver, who may answer "No": as
    # a statement it was printed as a fact (a three-person test, 25 Oct).
    from devops_transcription import clean_transcript

    assert clean_transcript("We use Kafka, right?") == "We use Kafka."
    cleaned = clean_transcript("Meera: So it stores invoices in MySQL, right? Arjun: No, it is PostgreSQL. "
                               "Meera: Okay, thanks.")
    assert "MySQL?" in cleaned and "right" not in cleaned
