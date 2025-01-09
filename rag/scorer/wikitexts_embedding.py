import json
import argparse
import numpy as np
from typing import List
from rag.faiss_manager import FAISSIndexManager
from rag.file_manager import FileManager
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from rag.scorer.document_scorer import DocumentScorer

class WikitextsDocumentScorer(DocumentScorer):
    def __init__(self, model="text-embedding-3-large"):
        self.model = model
        self.faiss_manager = FAISSIndexManager()

    def create_embedding(self, file_path):
        file_manager = FileManager(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
        # Load the file content as a dictionary
            data = json.load(f)
            documents = []
            for title, texts in data.items():
                # Create embeddings for each text
                for text in texts:
                    doc = Document(page_content=text, metadata={'source': title})
                    documents.append(doc)
            file_manager.dump_documents(documents)
        self.faiss_manager.upsert_file_to_faiss(file_manager, self.model)

    def score(self, claim: str, retrived_docs: List[Document]):
        total_score = 0
        for doc in retrived_docs:
            parsed_doc = self.faiss_manager.parse_result(doc)
            claim_embedding = self.faiss_manager.openaiManager.client.embeddings.create(input=[claim], model=self.model)
            claim_vector = np.array(claim_embedding.data[0].embedding).astype('float32').reshape(1, -1)
            doc_embedding = self.faiss_manager.index.reconstruct(parsed_doc['indice'])
            total_score += parsed_doc["score"] * cosine_similarity(claim_vector, doc_embedding.reshape(1, -1))
        return 0 if len(retrived_docs) == 0 else total_score / len(retrived_docs)

#python -m rag.scorer.wikitexts_embedding --file_path 'index_store/magazine/title_text_map.txt'
def main():
    parser = argparse.ArgumentParser(description="transfer wiki text into embedding format.")
    parser.add_argument("--file_path", required=True, type=str, help="path to the wiki text file")
    args = parser.parse_args()

    wikitexts_embedding = WikitextsDocumentScorer()
    wikitexts_embedding.create_embedding(args.file_path)
    query = "Which magazine was started first Arthur\'s Magazine or First for Women?"
    retrieved_docs = wikitexts_embedding.faiss_manager.search_faiss_index(query, top_k=10)
    print(retrieved_docs)
    response = wikitexts_embedding.faiss_manager.generate_response_from_context(query, retrieved_docs)
    print(response)
    print(wikitexts_embedding.score(query, retrieved_docs))

if __name__ == "__main__":
    print("Running wikitexts_embedding.py")
    main()