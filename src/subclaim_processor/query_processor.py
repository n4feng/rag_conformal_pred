from abc import ABC, abstractmethod

class IQueryProcessor(ABC):
    # take query file path output query with subclaims
    @abstractmethod
    def get_subclaims(self, query_file:str, subclaims_file: str):
        pass

    # add score into subclaim file
    @abstractmethod
    def score_subclaim(self, subclaim_file:str):
        pass

    # add annotation into subclaim file
    @abstractmethod
    def annotate_subclaim(self, subclaim_file:str):
        pass