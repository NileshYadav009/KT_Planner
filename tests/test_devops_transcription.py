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
