import os
import ast
import json
from jsonschema import RefResolver, validate

from src.common.retrieval import DocDB
from src.common.string_utils import extract_tag_content
from src.data_processor.raw_data_processor import IRawDataProcessor

class QueryProcessor(IRawDataProcessor):
    def __init__(self, db_path: str = ""):
        self.db = DocDB(db_path = db_path, data_path = None)
    
    def get_queries(self, dataset: str, input_file: str, output_file: str):
        """
        Reads raw data from a CSV file and extracts queries, storing them in a JSON file.
        """
        self.dataset = dataset

        if os.path.exists(output_file):
            print(f"{output_file} already exists.")
            return
        
        if dataset in ["fact_score", "hotpot_qa", "pop_qa"]:
            queries = []
            with open(input_file, 'r', encoding='utf-8') as file:
            
                for line in file:
                    #get base_schema from base_schema.json
                    if dataset == "fact_score":
                        title = line.strip()
                        query = {"input": f"What is {title}'s occupation?", 
                                "output": {"answer": "", "provenance": [{"title": title}]}}
                    elif dataset == "hotpot_qa":
                        data = json.loads(line)
                        query = {"input": data['input'], 
                                "output": {
                                        "answer": data['output'][0]['answer'],
                                        "provenance": [{"title": item['title'],
                                                        "wikipedia_id": int(item['wikipedia_id']),} for item in data['output'][0]['provenance']]}}
                    elif dataset == "pop_qa":
                        data = json.loads(line)
                        query = {"input": data['question'], 
                                "output": {
                                        "answer": ', or '.join(ast.literal_eval(data['possible_answers'])), 
                                        "provenance": [{"title": data['s_wiki_title'],
                                                        "wikipedia_id": data['subj_id'],}]}}
                
                with open(output_file, 'w', encoding='utf-8') as jsonfile:
                    json.dump(queries, jsonfile, indent=4)

        elif dataset == "medlfqa":
            all_queries = []

            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for file in os.listdir(input_file):
                if file.endswith('.json'):
                    with open(f"{input_file}/{file}", 'r', encoding='utf-8') as jsonfile:
                        data = json.load(jsonfile)
                        for item in data:
                            query = {
                                "input": item['Question'],
                                "output": {
                                    "answer": item['Must_have'],
                                    "provenance": [{"title": item['Question']}]}
                            }
                            all_queries.append(query)

            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(all_queries, jsonfile, indent=4)

    def get_documents(self, query_file: str, output_file: str):
        """
        Reads structured query data from a JSON file and generates a corresponding document list.
        """
        if os.path.exists(output_file):
            print(f"{output_file} already exists.")
            return

        base_schema = None
        wiki_schema = None
        with open("data/processed/base_schema.json", 'r', encoding='utf-8') as schemafile:
            base_schema = json.load(schemafile)

        with open("data/processed/wiki_schema.json", 'r', encoding='utf-8') as schemafile:
            wiki_schema = json.load(schemafile)
        
        if self.dataset in ["fact_score", "hotpot_qa", "pop_qa"]:
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
        elif self.dataset == "medlfqa":
            
            datasets = {}
            input_file = "data/raw/MedLFQA"
            for file in os.listdir(input_file):
                if file.endswith('.json'):
                    print(f"{input_file}/{file}")
                    with open(f"{input_file}/{file}", 'r', encoding='utf-8') as jsonfile:
                        datasets[file] = json.load(jsonfile)
            
            documents = {}
            for name, dataset in datasets.items():
                for pt in dataset:
                    pt_docs = []
                    pt_docs.extend([item.strip() for item in pt["Free_form_answer"].rstrip('.').split('.')])
                    pt_docs.extend(pt['Nice_to_have'])
                    documents[pt["Question"]] = pt_docs
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=4, ensure_ascii=False)

        print(f"Document list saved to {output_file}.")
        

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
    # processor = QueryProcessor("data/raw/enwiki-20230401.db")
    # processor.get_queries("fact_score", "data/raw/FactScore/factscore_names.txt", "data/processed/FactScore/fact_score_queries.json")
    # processor.get_documents("data/processed/FactScore/fact_score_queries.json", "data/processed/FactScore/title_text_map_factscore.txt")
    
    # processor = QueryProcessor("data/raw/enwiki-20230401.db")
    # processor.get_queries("hotpot_qa", "data/raw/HotpotQA/hotpotqa-dev-kilt-500.jsonl", "data/processed/HotpotQA/hotpotqa_queries.json")
    # processor.get_documents("data/processed/HotpotQA/hotpotqa_queries.json", "data/processed/HotpotQA/title_text_map_hotpotqa.txt")
    
    # processor = QueryProcessor("data/raw/enwiki-20230401.db")
    # processor.get_queries("pop_qa", "data/raw/PopQA/popQA_test.json", "data/processed/PopQA/popqa_queries.json")
    # processor.get_documents("data/processed/PopQA/popqa_queries.json", "data/processed/PopQA/title_text_map_popqa.txt")

    processor = QueryProcessor("data/raw/enwiki-20230401.db")
    processor.get_queries("medlfqa", "data/raw/MedLFQA/", "data/processed/MedLFQA/medlf_qa_queries.json")
    processor.get_documents("data/processed/MedLFQA/medlf_qa_queries.json", "data/processed/MedLFQA/title_text_map_medlfqa.txt")