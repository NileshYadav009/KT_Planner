"""Deployment steps (renderers/sections/deployment._steps)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderers.sections.deployment import _steps  # noqa: E402


def test_a_chain_of_steps_in_one_sentence_is_one_step_each_word_for_word():
    """X61: "GitHub Actions runs tests, Amazon ECR stores images, and ArgoCD
    deploys to EKS" was one step. The steps keep every word, the joining
    "and" included, so together they are the sentence as said."""
    said = "GitHub Actions runs tests, Amazon ECR stores images, and ArgoCD deploys to EKS."
    steps = _steps(said + " Rollback is a git revert.")
    assert steps == ["GitHub Actions runs tests,", "Amazon ECR stores images,", "and ArgoCD deploys to EKS.",
                     "Rollback is a git revert."]
    assert " ".join(steps[:3]) == said
    # Two clauses are not a chain.
    assert _steps("We deploy with Argo CD, and the rollback is a git revert.") == [
        "We deploy with Argo CD, and the rollback is a git revert."]
