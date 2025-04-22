import json

from src.rag.retrieval import DocDB
from src.utils.string_utils import extract_tag_content
from src.data_processor.raw_data_processor import DatasetProcessor


class DragonballProcessor(DatasetProcessor):
    def process_queries(self, input_file: str) -> list:
        # Preprocess Queries
        with open(input_file, "r", encoding="utf-8") as file:
            query_data = [json.loads(line) for line in file]

        queries = []
        for d in query_data:
            question = d["query"]
            gt_answer = d["ground_truth"]
            query = {
                "input": question["content"],
                "output": {
                    "answer": gt_answer["content"],
                    "provenance": [
                        {
                            "id": gt_answer["doc_ids"][i],
                            #  "title": gt_answer["doc_ids"]
                        }
                        for i in range(len(gt_answer["doc_ids"]))
                    ],
                },
                "metadata": {
                    "domain": d["domain"],
                    "query_type": question["query_type"],
                },
            }
            queries.append(query)
        return queries

    def process_documents(
        self, query_file: str, db: DocDB, queries: dict = None, **kwargs
    ) -> dict:
        if queries is None:
            with open(query_file, "r", encoding="utf-8") as jsonfile:
                queries = json.load(jsonfile)

        doc_file = "data/.source_data/Dragonball/dragonball_docs.jsonl"
        with open(doc_file, "r", encoding="utf-8") as file:
            docs_data = [json.loads(line) for line in file]

        corpus = {}
        for doc in docs_data:
            doc_data_dict = {
                # "doc_id": doc["doc_id"],
                "text": doc["content"],
                "doc_title": next(
                    (value for key, value in doc.items() if key.endswith("_name")), None
                ),
                "metadata": {"domain": doc["domain"]},
                "text_chunks": [{"para_id": 0, "text": doc["content"]}],
            }
            corpus[doc["doc_id"]] = doc_data_dict

        documents = {}
        for query in queries:
            for provenance in query["output"]["provenance"]:
                id = provenance["id"]
                documents[id] = corpus[id]["text"]
        return documents
