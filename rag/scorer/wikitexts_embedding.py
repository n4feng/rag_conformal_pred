import json
import argparse
import numpy as np
import warnings
from typing import List
from collections import defaultdict
from rag.faiss_manager import FAISSIndexManager
from rag.file_manager import FileManager
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from rag.scorer.document_scorer import DocumentScorer
from rag.llm.openai_atomicfact_generator import OpenAIAtomicFactGenerator

class WikitextsDocumentScorer(DocumentScorer):
    def __init__(self, model="text-embedding-3-large"):
        self.model = model
        self.faiss_manager = FAISSIndexManager()
        self.gen = OpenAIAtomicFactGenerator()

    def query_model(self, prompt, model, max_tokens=1000, temperature=0, n_samples=1):
        messages = [{"role": "user", "content": prompt}]
        completion = self.faiss_manager.openaiManager.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n_samples,
        )
        return completion.choices[0].message.content

    def create_embedding(self, file_path):
        file_manager = FileManager(file_path)
        #only process the file and dump the documents if ..._texts.json metadata file is not created
        if(not file_manager.texts):
            with open(file_path, 'r', encoding='utf-8') as f:
            # Load the file content as a dictionary
                data = json.load(f)
                documents = []
                for title, texts in data.items():
                    # Create embeddings for each text
                    for text in texts:
                        doc = Document(page_content=title+": "+text, metadata={'source': title, 'file_path': file_path})
                        documents.append(doc)
                file_manager.dump_documents(documents)
        self.faiss_manager.upsert_file_to_faiss(file_manager, self.model)

    def score(self, claim: str, retrived_docs: List[Document]):
        # claim score will be the maximum product of cosine similarity between the claim and the retrieved documents
        doc_scores = []
        for doc in retrived_docs:
            parsed_doc = self.faiss_manager.parse_result(doc)
            claim_embedding = self.faiss_manager.openaiManager.client.embeddings.create(input=[claim], model=self.model)
            claim_vector = np.array(claim_embedding.data[0].embedding).astype('float32').reshape(1, -1)
            doc_embedding = self.faiss_manager.index.reconstruct(parsed_doc['indice'])
            doc_scores.append(parsed_doc["score"] * cosine_similarity(claim_vector, doc_embedding.reshape(1, -1))[0][0])
        return 0 if len(retrived_docs) == 0 else max(doc_scores)
    
    def say_less(self, prompt, thresholds, model='gpt-4'):
        """
        say_less takes in the model prompt, generate output, breaks it down into subclaims, and removes sub-claims up to the threshold value.
        """
        output = ""
        retrieved_docs = self.faiss_manager.search_faiss_index(prompt, 10, 0.3)
        output = self.faiss_manager.generate_response_from_context(prompt, retrieved_docs)
        atomicFacts = self.gen.get_facts_from_text(output)
        subclaims_with_score = []
        for fact in atomicFacts:
            purefact = fact.rpartition(':')[0] if ':' in fact else fact
            score = self.score(purefact, retrieved_docs)
            #store purefact and score pair into list
            subclaims_with_score.append((purefact, score))
        accepted_subclaims_per_threshold = []
        mergerd_output_per_threshold = []
        for threshold in thresholds:
            accepted_subclaims = [subclaim for subclaim in subclaims_with_score if subclaim[1] > threshold]
            mergerd_output = self.merge_subclaims(accepted_subclaims, model, prompt)
            accepted_subclaims_per_threshold.append(accepted_subclaims)
            mergerd_output_per_threshold.append(mergerd_output)
        return (output, mergerd_output_per_threshold, subclaims_with_score, accepted_subclaims_per_threshold)

    #!!!!!!!!!!!!!!!Below is a wrong way to do conditional as it is not doing any re-calibration       
    # def say_less_conditional(self, prompt, thresholds_path, model='gpt-4'):
    #     """
    #     compute the threshold based on retrived document type and weight of document in each type
    #     """
    #     conditional_threshold = self.calibrate_conditional_threshold(prompt, thresholds_path)
    #     return self.say_less(prompt, conditional_threshold, model)


    # def calibrate_conditional_threshold(self, prompt, thresholds_path):
    #     retrieved_docs = self.faiss_manager.search_faiss_index(prompt, 10, 0.3)
    #     parsed_docs = [self.faiss_manager.parse_result(doc) for doc in retrieved_docs]
    #     #count each filepath in metadata
    #     doc_count = defaultdict(int)
    #     for doc in parsed_docs:  
    #         doc_count[doc['metadata']['file_path']] += 1
    #     total_docs = len(parsed_docs)
    #     #get the file name have the highest count
    #     max = 0
    #     max_name = ""
    #     for key, val in doc_count.items():
    #         if val > max:
    #             max = val
    #             max_name = key
        
    #     with open(thresholds_path, 'r') as f:
    #         data = json.load(f)
    #     for count_key, count_val in doc_count.items():
    #         match_found = False  # Flag to track if a match is found
    #         for key, val in data.items():
    #             if key in max_name:
    #                 conditional_threshold = val
    #                 match_found = True  # Set flag to True if a match is found
    #                 break  # Exit the inner loop since the match is found
    #         if not match_found:  # If no match was found after checking all count_keys
    #             data_keys = data.keys()
    #             warnings.warn(
    #                 f"File name {count_key} not found in the precalculated threshold set {data_keys}",
    #                 UserWarning
    #             )
    #             conditional_threshold = 1.0 #if not find, set largest threshold so no claims will remain
    #     #print(f"Conditional threshold now is: {conditional_threshold}")
    #     return conditional_threshold
    
    def default_merge_prompt( subclaims, prompt):
        claim_string = "\n".join(
            [str(i) + ": " + subclaim[0] for i, subclaim in enumerate(subclaims)]
        )
        return f"You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and try to use all facts as long as its possible. If no facts are given, reply to the instruction incorporating the fact that you dont know enough to fully respond. \n\nThe facts:\n{claim_string}\n\nThe instruction:\n{prompt}"

    def merge_subclaims(
        self, subclaims, model, prompt, create_merge_prompt=default_merge_prompt
    ):
        """
        Takes in a list of sub-claims like [('Percy Liang is a computer scientist.', 5.0), ...] and produces a merged output.
        """
        prompt = create_merge_prompt(subclaims, prompt)
        output = (
            self.query_model(prompt, model, max_tokens=1000, temperature=0)
            if subclaims
            else "Abstain."
        )
        return output


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