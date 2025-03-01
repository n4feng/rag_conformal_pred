import ast
import json

from jsonschema import RefResolver, validate

from src.datacleaning.raw_data_processor import RawDataProcessor
from src.common.retrieval import DocDB
from src.common.string_utils import extract_tag_content

class PopQAProcessor(RawDataProcessor):
    def __init__(self, db_path: str = ""):
        self.db = DocDB(db_path = db_path, data_path = None)
    
    def get_queries(self, input_file: str, output_file: str):
        """
        Reads raw data from a json file and extracts queries, storing them in a JSON file.
        """
        queries = []
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                # get base_schema from base_schema.json
                query = {"input": data['question'], 
                            "output": {
                                    "answer": ', or '.join(ast.literal_eval(data['possible_answers'])), 
                                    "provenance": [{"title": data['s_wiki_title'],
                                                    "wikipedia_id": data['subj_id'],}]}}
                queries.append(query)
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(queries, jsonfile, indent=4)

    def get_documents(self, query_file: str, output_file: str):
        """
        Reads structured query data from a JSON file and generates a corresponding document list.
        """
        base_schema = None
        wiki_schema = None
        with open("data/processed/base_schema.json", 'r', encoding='utf-8') as schemafile:
            base_schema = json.load(schemafile)

        with open("data/processed/wiki_schema.json", 'r', encoding='utf-8') as schemafile:
            wiki_schema = json.load(schemafile)
        
        documents = {}
        with open(query_file, 'r', encoding='utf-8') as jsonfile:
            queries = json.load(jsonfile)
            for query in queries:
                resolver = RefResolver("data/processed/base_schema.json", base_schema)
                validate(instance=query, schema=wiki_schema, resolver=resolver)
                title = query["output"]["provenance"][0]["title"]
                document = self.get_documents_per_query(title)
                documents[title] = document
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(documents, jsonfile, indent=4)
        

    def get_documents_per_query(self, title) -> list:
        """
        Returns a list of documents for a given query.
        """
        contents = ""
        try:
            docs = self.db.get_text_from_title(title)
            for data in docs:
                contents += data["text"]
            return extract_tag_content(contents)
        except Exception as e:
            print(f"Error retrieving documents for title {title}: {e}")
            return []
        

# Example usage
if __name__ == "__main__":
    processor = PopQAProcessor("data/raw/enwiki-20230401.db")
    processor.get_queries("data/raw/PopQA/popQA_test.json", "data/processed/PopQA/popqa_queries.json")
    processor.get_documents("data/processed/PopQA/popqa_queries.json", "data/processed/PopQA/title_text_map_popqa.txt")