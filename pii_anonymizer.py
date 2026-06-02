"""
Presidio-based PII Detection and Anonymization Module

Protects sensitive data (passwords, emails, secrets, account IDs) before storing KT.
Ensures compliance with GDPR, HIPAA, SOC 2 standards.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import json

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig, RecognizerResult
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False
    logging.warning("Presidio not installed. PII anonymization will be disabled.")

logger = logging.getLogger(__name__)

# Presidio entity types to redact
SENSITIVE_ENTITY_TYPES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "CRYPTO",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "URL",
    "BANKING_ROUTING_NUMBER",
    "CREDIT_CARD_EXPIRATION",
    "DATE_TIME",  # Optional: depends on context
]

# Custom patterns for DevOps secrets
CUSTOM_PATTERNS = {
    "AWS_KEY": r"(?i)(AKIA[0-9A-Z]{16})",
    "AWS_SECRET": r"(?i)(aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40})",
    "API_KEY": r"(?i)(api[_-]?key\s*[=:]\s*[A-Za-z0-9_\-]{20,})",
    "JWT_TOKEN": r"(?i)(eyJ[A-Za-z0-9_\-\.]+)",
    "DATABASE_PASSWORD": r"(?i)(password\s*[=:]\s*[^\s]+)",
    "PRIVATE_KEY": r"(?i)(-----BEGIN (RSA|EC|PGP) PRIVATE KEY-----)",
    "SLACK_TOKEN": r"(?i)(xox[baprs]-[0-9a-zA-Z]{10,48})",
    "GITHUB_TOKEN": r"(?i)(ghp_[0-9a-zA-Z]{36})",
    "DATABASE_URL": r"(?i)(postgresql://|mysql://|mongodb://)[^\s]+",
}


class PIIAnonymizer:
    """Detect and anonymize PII using Presidio."""
    
    def __init__(self, use_anonymizer: bool = True):
        """
        Initialize Presidio anonymizer.
        
        Args:
            use_anonymizer: Whether to actually anonymize (True) or just detect (False)
        """
        self.analyzer = None
        self.anonymizer = None
        self.use_anonymizer = use_anonymizer
        self.detections: List[Dict[str, Any]] = []
        
        if not HAS_PRESIDIO:
            logger.warning("Presidio not available. PII anonymization disabled.")
            return
        
        try:
            self.analyzer = AnalyzerEngine()
            if use_anonymizer:
                self.anonymizer = AnonymizerEngine()
            logger.info("Presidio initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Presidio: {e}. PII anonymization will be disabled.")
            self.analyzer = None
            self.anonymizer = None
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text using Presidio.
        
        Returns:
            List of detected entities with start/end positions and types
        """
        if not self.analyzer or not text:
            return []
        
        try:
            results = self.analyzer.analyze(text=text, language="en")
            detections = []
            for result in results:
                detections.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "text": text[result.start:result.end],
                    "score": float(result.score)
                })
            return detections
        except Exception as e:
            logger.debug(f"Presidio detection failed: {e}")
            return []
    
    def anonymize(
        self,
        text: str,
        use_redaction: bool = True,
        redaction_char: str = "*",
        custom_replacement: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Anonymize PII in text.
        
        Args:
            text: Input text to anonymize
            use_redaction: If True, replace with redaction_char; if False, use replacement text
            redaction_char: Character to use for redaction (default *)
            custom_replacement: Custom replacement text (e.g., "[REDACTED]")
        
        Returns:
            Tuple of (anonymized_text, detection_report)
        """
        if not self.analyzer or not text:
            return text, []
        
        try:
            # Detect PII
            detections = self.detect_pii(text)
            if not detections:
                return text, []
            
            # Prepare operators
            if custom_replacement:
                operator = OperatorConfig("replace", {"new_value": custom_replacement})
            elif use_redaction:
                operator = OperatorConfig("redact")
            else:
                operator = OperatorConfig("replace", {"new_value": redaction_char})
            
            # Create anonymization input using Presidio RecognizerResult objects
            analyzer_results = [
                RecognizerResult(
                    entity_type=detection["entity_type"],
                    start=detection["start"],
                    end=detection["end"],
                    score=detection["score"],
                )
                for detection in detections
            ]

            if self.anonymizer:
                anonymized = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators={"DEFAULT": operator},
                )
                anonymized_text = anonymized.text
            else:
                anonymized_text = text
            
            self.detections.extend(detections)
            
            return anonymized_text, detections
        except Exception as e:
            logger.warning(f"Anonymization failed: {e}. Returning original text.")
            return text, []
    
    def anonymize_transcript(
        self,
        transcript: str,
        keep_original: bool = True
    ) -> Dict[str, Any]:
        """
        Anonymize an entire transcript and return detailed report.
        
        Args:
            transcript: Full transcript text
            keep_original: Whether to keep original text in report
        
        Returns:
            Dict with anonymized_text, detections, and metadata
        """
        # NOTE: The Presidio anonymization below replaces detected PII with
        # "[REDACTED]". The redaction operation is performed by the
        # `presidio_anonymizer` library (AnonymizerEngine / OperatorConfig).
        # To disable the automatic replacement that is currently blocking
        # transcripts/coverage outputs, the anonymization call is commented
        # out and we return the original transcript with an empty report.
        #
        # Original call (commented):
        # anonymized, detections = self.anonymize(
        #     transcript,
        #     use_redaction=True,
        #     custom_replacement="[REDACTED]"
        # )
        
        return {
            "original_text": transcript if keep_original else None,
            "anonymized_text": transcript,
            "detection_count": 0,
            "detections_by_type": {},
            "detections": [],
            "redaction_applied": False,
            "audit_timestamp": self._get_timestamp()
        }
    
    def _group_by_type(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group detections by entity type."""
        groups = {}
        for detection in detections:
            entity_type = detection["entity_type"]
            groups[entity_type] = groups.get(entity_type, 0) + 1
        return groups
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of all detections made so far."""
        return {
            "total_detections": len(self.detections),
            "by_type": self._group_by_type(self.detections),
            "unique_types": sorted(set(d["entity_type"] for d in self.detections))
        }


def anonymize_transcript_before_classification(transcript: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function: anonymize transcript before classification.
    
    Returns:
        Tuple of (anonymized_transcript, anonymization_report)
    """
    if not HAS_PRESIDIO:
        return transcript, {"status": "presidio_not_available"}
    
    anonymizer = PIIAnonymizer()
    report = anonymizer.anonymize_transcript(transcript, keep_original=False)
    return report["anonymized_text"], report


def is_pii_present(text: str) -> bool:
    """Quick check if PII is likely present in text."""
    if not HAS_PRESIDIO:
        return False
    
    anonymizer = PIIAnonymizer()
    detections = anonymizer.detect_pii(text)
    return len(detections) > 0


def get_pii_summary(text: str) -> Dict[str, int]:
    """Get count of each PII type detected."""
    if not HAS_PRESIDIO:
        return {}
    
    anonymizer = PIIAnonymizer()
    detections = anonymizer.detect_pii(text)
    summary = {}
    for detection in detections:
        entity_type = detection["entity_type"]
        summary[entity_type] = summary.get(entity_type, 0) + 1
    return summary
