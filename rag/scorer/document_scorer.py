from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class DocumentScorer(ABC):
    @abstractmethod
    def create_embedding(self, file_path: str):
        pass

    @abstractmethod
    def score(self, claim: str, retrieved_docs: List[Document]) -> float:
        pass