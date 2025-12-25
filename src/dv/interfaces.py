from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.dv.models.entities import Chunk, ValuationResult

class Valuator(ABC):
    @abstractmethod
    def evaluate(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        """Calculates value for each chunk given the query and answer."""
        pass

class Signaler(ABC):
    @abstractmethod
    def get_signals(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """Extracts model-internal signals (attention, gradients)."""
        pass

class Judge(ABC):
    @abstractmethod
    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        """Returns a 0.0-1.0 faithfulness score."""
        pass
