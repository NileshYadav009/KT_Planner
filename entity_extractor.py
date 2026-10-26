"""
GLiNER-based Entity Extraction for KT Classification

Extracts structured entities (tools, environments, owners, escalation levels, services)
from sentences to enrich classification with metadata.
"""

from collections import OrderedDict
from typing import Dict, List, Any, Optional
import logging
import threading

try:
    from gliner import GLiNER
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False
    logging.warning("GLiNER not installed. Entity extraction will be disabled.")

logger = logging.getLogger(__name__)


# Texts whose entities are kept (EntityExtractor.prefetch, _predict).
MEMO_TEXTS = 5_000

# Default entity types to extract
DEFAULT_ENTITY_LABELS = [
    "tool",           # Terraform, Jenkins, Kubernetes, etc.
    "environment",    # prod, staging, dev, production
    "service",        # Aurora, RDS, Lambda, S3, etc.
    "owner",          # Team names, person names
    "escalation",     # Critical, P1, high risk, urgent
    "technology",     # Docker, Python, Java, etc.
    "platform",       # AWS, GCP, Azure, Kubernetes
    "database",       # MySQL, PostgreSQL, MongoDB, etc.
    "monitoring",     # DataDog, Prometheus, CloudWatch, etc.
]


class EntityExtractor:
    """Extract structured entities from sentences using GLiNER."""
    
    def __init__(
        self,
        model_name: str = "urchade/gliner_multi",
        entity_labels: Optional[List[str]] = None,
        use_gliner: bool = True
    ):
        """
        Initialize GLiNER entity extractor.
        
        Args:
            model_name: GLiNER model to use
            entity_labels: List of entity types to extract
            use_gliner: Whether to actually load GLiNER (False for testing)
        """
        self.model = None
        self.entity_labels = entity_labels or DEFAULT_ENTITY_LABELS
        self._memo: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self._memo_lock = threading.Lock()
        self.use_gliner = use_gliner
        
        if not _GLINER_AVAILABLE or not use_gliner:
            logger.info("Entity extraction disabled (GLiNER not available or disabled)")
            return
        
        try:
            self.model = GLiNER.from_pretrained(model_name)
            logger.info(f"Loaded GLiNER model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load GLiNER model: {e}. Entity extraction will be disabled.")
            self.model = None
    
    def prefetch(self, texts: List[str]) -> None:
        """Predict the entities of every text not yet known, eight at a time:
        GLiNER one fact at a time was about a third of a KT's time on a CPU,
        and in batches it takes half as long, with the same entities (all 57
        facts of three KTs, scores within 1e-5)."""
        if not self.model:
            return
        with self._memo_lock:
            todo = list(dict.fromkeys(t for t in texts if t and t.strip() and t not in self._memo))
        if not todo:
            return
        try:
            results = self.model.inference(todo, self.entity_labels, batch_size=8)
        except Exception as e:
            logger.debug(f"Batched entity extraction failed, texts left to one at a time: {e}")
            return
        self._remember(dict(zip(todo, results)))

    def _predict(self, text: str) -> List[Dict[str, Any]]:
        """GLiNER's entities for `text`, from what prefetch() or an earlier
        call found where it can. Results are not changed by callers, so the
        same list is returned."""
        with self._memo_lock:
            if text in self._memo:
                self._memo.move_to_end(text)
                return self._memo[text]
        entities = self.model.predict_entities(text, self.entity_labels)
        self._remember({text: entities})
        return entities

    def _remember(self, found: Dict[str, List[Dict[str, Any]]]) -> None:
        with self._memo_lock:
            self._memo.update(found)
            while len(self._memo) > MEMO_TEXTS:
                self._memo.popitem(last=False)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.
        
        Returns:
            Dict mapping entity type to list of extracted values
            Example: {
                "tool": ["Terraform", "Jenkins"],
                "environment": ["prod", "staging"],
                "escalation": ["critical"]
            }
        """
        if not self.model or not text or not text.strip():
            return {}
        
        try:
            entities = self._predict(text)
            
            # Organize by entity type
            result: Dict[str, List[str]] = {}
            for entity in entities:
                entity_type = entity.get("label", "").lower()
                entity_text = entity.get("text", "").strip()
                
                if entity_type and entity_text:
                    if entity_type not in result:
                        result[entity_type] = []
                    # Avoid duplicates
                    if entity_text not in result[entity_type]:
                        result[entity_type].append(entity_text)
            
            return result
        except Exception as e:
            logger.debug(f"Entity extraction failed for text: {text[:50]}... Error: {e}")
            return {}
    
    def extract_with_confidence(
        self,
        text: str,
        min_confidence: float = 0.3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities with confidence scores.
        
        Returns:
            Dict mapping entity type to list of dicts with 'text' and 'confidence'
        """
        if not self.model or not text or not text.strip():
            return {}
        
        try:
            entities = self._predict(text)
            
            # Organize by entity type with confidence
            result: Dict[str, List[Dict[str, Any]]] = {}
            seen = set()  # Track (entity_type, entity_text) pairs to avoid duplicates
            
            for entity in entities:
                entity_type = entity.get("label", "").lower()
                entity_text = entity.get("text", "").strip()
                confidence = entity.get("score", 0.0)
                
                if entity_type and entity_text and confidence >= min_confidence:
                    key = (entity_type, entity_text.lower())
                    if key not in seen:
                        if entity_type not in result:
                            result[entity_type] = []
                        result[entity_type].append({
                            "text": entity_text,
                            "confidence": float(confidence)
                        })
                        seen.add(key)
            
            return result
        except Exception as e:
            logger.debug(f"Confidence-based entity extraction failed: {e}")
            return {}
    
    def get_context_entities(
        self,
        text: str,
        max_entities_per_type: int = 3
    ) -> Dict[str, List[str]]:
        """
        Extract key entities for use as classification context.
        Limits to top N entities per type to avoid noise.
        """
        entities_with_conf = self.extract_with_confidence(text)
        
        result: Dict[str, List[str]] = {}
        for entity_type, entities_list in entities_with_conf.items():
            # Sort by confidence and take top N
            sorted_entities = sorted(
                entities_list,
                key=lambda x: x.get("confidence", 0.0),
                reverse=True
            )
            result[entity_type] = [
                e["text"] for e in sorted_entities[:max_entities_per_type]
            ]
        
        return result
