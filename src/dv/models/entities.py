from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
import uuid

class ValuationMethod(str, Enum):
    LOO = "LOO"
    SHAPLEY = "SHAPLEY"
    ATTENTION = "ATTENTION"

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValuationResult:
    chunk_id: str
    method: ValuationMethod
    score: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExperimentRun:
    query: str
    retrieved_chunks: List[Chunk]
    generated_answer: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ground_truth_faithfulness: Optional[float] = None
    valuation_reports: List[ValuationResult] = field(default_factory=list)
