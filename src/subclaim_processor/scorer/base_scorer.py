from abc import ABC, abstractmethod

class IScorer(ABC):
    @abstractmethod
    def score(self, claim: str) -> float:
        pass

    def get_type(self):
        return self.__class__.__name__.replace("Scorer", "").lower()
