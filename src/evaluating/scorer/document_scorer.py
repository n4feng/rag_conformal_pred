from abc import abstractmethod
from typing import List
from langchain.schema import Document
from src.evaluating.scorer.base_scorer import IScorer

class IDocumentScorer(IScorer):

    @abstractmethod
    def score(self, claim: str, retrieved_docs: List[Document]) -> float:
        pass
