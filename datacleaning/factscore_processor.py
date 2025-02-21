from datacleaning.raw_data_processor import RawDataProcessor
from common.string_utils import extract_tag_content
import json
from jsonschema import validate
from common.retrieval import DocDB

class FactScoreProcessor(RawDataProcessor):
    def __init__(self, db_path: str):
        self.db = DocDB(db_path = db_path, data_path = None)
    
    def get_queries(self, input_file: str, output_file: str):
        """
        Reads raw data from a CSV file and extracts queries, storing them in a JSON file.
        """
        queries = []
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                #get base_schema from base_schema.json
                title = line.strip()
                query = {"input": f"What is {title}'s occupation?", 
                         "output": {"answer": "", "provenance": [{"title": title}]}}
                queries.append(query)
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(queries, jsonfile, indent=4)

    def get_documents(self, query_file: str, output_file: str):
        """
        Reads structured query data from a JSON file and generates a corresponding document list.
        """
        base_schema = None
        with open("data/processed/base_schema.json", 'r', encoding='utf-8') as schemafile:
            base_schema = json.load(schemafile)
        
        documents = {}
        with open(query_file, 'r', encoding='utf-8') as jsonfile:
            queries = json.load(jsonfile)
            for query in queries:
                validate(instance=query, schema=base_schema)
                title = query["output"]["provenance"][0]["title"]
                document = self.get_documents_per_query(title)
                documents[title] = document
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(documents, jsonfile, indent=4)
        

    def get_documents_per_query(self, title) -> list:
        """
        Returns a list of documents for a given query.
        """
        docs = self.db.get_text_from_title(title)
        contents = ""
        for data in docs:
            contents += (data["text"])
        return extract_tag_content(contents)

# Example usage
if __name__ == "__main__":
    processor = FactScoreProcessor("data/raw/enwiki-20230401.db")
    processor.get_queries("data/raw/factscore_names.txt", "data/processed/fact_score_queries.json")
    processor.get_documents("data/processed/fact_score_queries.json", "data/processed/title_text_map_factscore.txt")