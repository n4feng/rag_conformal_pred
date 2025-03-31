import json
import argparse
import numpy as np
from typing import List
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from src.common.faiss_manager import FAISSIndexManager
from src.common.file_manager import FileManager
from src.common.llm.openai_atomicfact_generator import OpenAIAtomicFactGenerator
from src.subclaim_processor.scorer.document_scorer import IDocumentScorer


class SimilarityScorer(IDocumentScorer):
    def __init__(
        self,
        embedding_model="text-embedding-3-large",
        index_path="index_store/index.faiss",
        indice2fm_path="index_store/indice2fm.json",
    ):
        self.embedding_model = embedding_model
        self.faiss_manager = FAISSIndexManager(
            index_path=index_path, indice2fm_path=indice2fm_path
        )
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

    def score(self, claim: str, retrived_docs: List[Document]):
        # claim score will be the maximum product of cosine similarity between the claim and the retrieved documents
        doc_scores = []
        for doc in retrived_docs:
            parsed_doc = self.faiss_manager.parse_result(doc)
            claim_embedding = self.faiss_manager.openaiManager.client.embeddings.create(
                input=[claim], model=self.embedding_model
            )
            claim_vector = (
                np.array(claim_embedding.data[0].embedding)
                .astype("float32")
                .reshape(1, -1)
            )
            doc_embedding = self.faiss_manager.index.reconstruct(parsed_doc["indice"])
            doc_scores.append(
                parsed_doc["score"]
                * cosine_similarity(claim_vector, doc_embedding.reshape(1, -1))[0][0]
            )
        return 0 if len(retrived_docs) == 0 else max(doc_scores)

    def say_less(self, prompt, thresholds, model="gpt-4"):
        """
        say_less takes in the model prompt, generate output, breaks it down into subclaims, and removes sub-claims up to the threshold value.
        """
        output = ""
        retrieved_docs = self.faiss_manager.search_faiss_index(prompt, 10, 0.3)
        output = self.faiss_manager.generate_response_from_context(
            prompt, retrieved_docs
        )
        atomicFacts = self.gen.get_facts_from_text(output)
        subclaims_with_score = []
        for fact in atomicFacts:
            purefact = fact.rpartition(":")[0] if ":" in fact else fact
            score = self.score(purefact, retrieved_docs)
            # store purefact and score pair into list
            subclaims_with_score.append((purefact, score))
        accepted_subclaims_per_threshold = []
        mergerd_output_per_threshold = []
        for threshold in thresholds:
            accepted_subclaims = [
                subclaim for subclaim in subclaims_with_score if subclaim[1] > threshold
            ]
            mergerd_output = self.merge_subclaims(accepted_subclaims, model, prompt)
            accepted_subclaims_per_threshold.append(accepted_subclaims)
            mergerd_output_per_threshold.append(mergerd_output)
        return (
            output,
            mergerd_output_per_threshold,
            subclaims_with_score,
            accepted_subclaims_per_threshold,
        )

    def default_merge_prompt(subclaims, prompt):
        claim_string = "\n".join(
            [str(i) + ": " + subclaim[0] for i, subclaim in enumerate(subclaims)]
        )
        return f"You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and try to use all facts as long as its possible. If no facts are given, reply to the instruction incorporating the fact that you don't know enough to fully respond. \n\nThe facts:\n{claim_string}\n\nThe instruction:\n{prompt}"

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


def main():
    parser = argparse.ArgumentParser(
        description="transfer wiki text into embedding format."
    )
    parser.add_argument(
        "--file_path", required=True, type=str, help="path to the wiki text file"
    )
    args = parser.parse_args()

    wikitexts_embedding = SimilarityScorer()
    wikitexts_embedding.create_embedding(args.file_path)
    query = "Which magazine was started first Arthur's Magazine or First for Women?"
    retrieved_docs = wikitexts_embedding.faiss_manager.search_faiss_index(
        query, top_k=10
    )
    print(retrieved_docs)
    response = wikitexts_embedding.faiss_manager.generate_response_from_context(
        query, retrieved_docs
    )
    print(response)
    print(wikitexts_embedding.score(query, retrieved_docs))


if __name__ == "__main__":
    print("Running wikitexts_embedding.py")
    main()
