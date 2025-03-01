from abc import ABC, abstractmethod

class RawDataProcessor(ABC):
    @abstractmethod
    # take input file path of raw data and output structured query data
    def get_queries(self, input_file:str, output_file:str):
        pass

    # take structured input query file path of structured query data and output raw data
    @abstractmethod
    def get_documents(self, query_file:str, output_file:str):
        pass

    
    @abstractmethod
    def get_documents_per_query(self, query) -> list:
        pass