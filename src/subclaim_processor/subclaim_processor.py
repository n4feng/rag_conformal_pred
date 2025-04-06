import os
import json
import random
from typing import Union, Optional
from tqdm import tqdm
from jsonschema import RefResolver, validate
from src.subclaim_processor.query_processor import IQueryProcessor
from src.common.llm.openai_rag_agent import OpenAIRAGAgent
from src.common.llm.openai_atomicfact_generator import OpenAIAtomicFactGenerator
from src.common.llm.openai_claim_verification import OpenAIClaimVerification
from src.subclaim_processor.scorer.base_scorer import IScorer
from src.subclaim_processor.scorer.document_scorer import IDocumentScorer
from src.subclaim_processor.scorer.subclaim_scorer import SubclaimScorer
from src.common.faiss_manager import FAISSIndexManager
from src.common.file_manager import FileManager
from src.calibration.utils import load_subclaim_data


class SubclaimProcessor(IQueryProcessor):
    def __init__(self, faiss_manager, scorer: IScorer, subclaims_file: str):
        self.faiss_manager = faiss_manager
        self.response_agent = OpenAIRAGAgent(faiss_manager)
        self.generator = OpenAIAtomicFactGenerator()
        self.verifier = OpenAIClaimVerification()
        self.scorer = scorer
        self.subclaims_file = subclaims_file
        with open(
            "data/out/subclaims_schema.json", "r", encoding="utf-8"
        ) as schemafile:
            self.subclaim_schema = json.load(schemafile)

    def get_subclaims(
        self,
        query_file: str,
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        truncate_by: Optional[str] = "\n",
    ):

        with open(query_file, "r", encoding="utf-8") as jsonfile:
            queries = json.load(jsonfile)

        with open(self.subclaims_file, "w", encoding="utf-8") as subclaimsfile:
            subclaimsfile.write("[\n")

        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            question = query["input"]

            # document retrieval
            retrieved_docs = self.faiss_manager.search_faiss_index(
                question,
                top_k=10,
                threshold=0.3,
                truncation_strategy=truncation_strategy,
                truncate_by=truncate_by,
            )

            chat_response = self.response_agent.answer(
                question, retrieved_docs, temperature=0.7, n_samples=1
            )
            response = chat_response.choices[0].message.content
            subclaims_with_log_probs = self.generator.get_facts_from_text(response)
            subclaims_entry = {
                "query": question,
                "gld_ans": query["output"]["answer"],
                "retrieved_docs": retrieved_docs,
                "response": response,
                "subclaims": [],
            }

            for subclaim in subclaims_with_log_probs:
                subclaims_entry["subclaims"].append(
                    {
                        "subclaim": subclaim[0],
                        "scores": {
                            "log_prob": [score for token, score in subclaim[1]]
                        },  # Add logic to populate scores if available
                        "annotations": {},  # Add logic to populate annotations if available
                    }
                )

            with open(self.subclaims_file, "a", encoding="utf-8") as subclaimsfile:
                json.dump(subclaims_entry, subclaimsfile, indent=4)
                if i < len(queries) - 1:
                    subclaimsfile.write(",\n")

        with open(self.subclaims_file, "a", encoding="utf-8") as subclaimsfile:
            subclaimsfile.write("\n]")
        print(f"Query responses saved to {self.subclaims_file}.")

    def score_subclaim(self):
        with open(self.subclaims_file, "r", encoding="utf-8") as jsonfile:
            subclaims_data = json.load(jsonfile)
            for entry in tqdm(subclaims_data, desc="Scoring subclaims"):
                validate(instance=entry, schema=self.subclaim_schema)
                for i, subclaim in enumerate(entry["subclaims"]):
                    score = 0.0
                    if isinstance(self.scorer, IDocumentScorer):
                        if "relavance" not in subclaim["scores"].keys():
                            if "conditional_similarity" in subclaim["scores"].keys():
                                subclaim["scores"]["relavance"] = subclaim[
                                    "scores"
                                ].pop("conditional_similarity")
                            elif "joint_similarity" in subclaim["scores"].keys():
                                subclaim["scores"]["relavance"] = subclaim[
                                    "scores"
                                ].pop("joint_similarity")
                            else:  # TODO
                                relavance_score = self.scorer.score(
                                    subclaim["subclaim"], entry["retrieved_docs"]
                                )
                                subclaim["scores"]["relavance"] = float(relavance_score)
                        if "cosine_similarity" not in subclaim["scores"].keys():
                            if "baseline" in subclaim["scores"].keys():
                                subclaim["scores"]["cosine_similarity"] = subclaim[
                                    "scores"
                                ].pop("baseline")
                            else:  # TODO
                                cosine_similarity_score = self.scorer.cosine_similarity(
                                    subclaim["subclaim"], entry["query"]
                                )
                                subclaim["scores"]["cosine_similarity"] = float(
                                    cosine_similarity_score
                                )
                        if "frequency" not in subclaim["scores"].keys():
                            frequency_score = self.scorer.frequency_score(
                                response_agent=self.response_agent,
                                question=entry["query"],
                                subclaim=subclaim["subclaim"],
                                retrived_docs=entry["retrieved_docs"],
                                temperature=1,
                                n_samples=5,
                            )
                            subclaim["scores"]["frequency"] = float(frequency_score)
                        if "random" not in subclaim["scores"].keys():
                            subclaim["scores"]["random"] = random.random()
                        if "ordinal" not in subclaim["scores"].keys():
                            subclaim["scores"]["ordinal"] = (
                                (i / len(entry["subclaims"]))
                                if len(entry["subclaims"]) > 0
                                else 0
                            )

        with open(self.subclaims_file, "w", encoding="utf-8") as jsonfile:
            json.dump(subclaims_data, jsonfile, indent=4)
        print(f"Subclaims with scores saved to {self.subclaims_file}.")

    def annotate_subclaim(self):
        with open(self.subclaims_file, "r", encoding="utf-8") as jsonfile:
            subclaims_data = json.load(jsonfile)
            for entry in tqdm(subclaims_data, desc="Annotating subclaims"):
                validate(instance=entry, schema=self.subclaim_schema)
                doc_contents = []
                for doc in entry["retrieved_docs"]:
                    try:
                        # Split the document string into page_content and metadata
                        doc_parts = doc.split("metadata=")
                        page_content = doc_parts[0].replace("page_content=", "").strip()
                        doc_contents.append(page_content)
                    except Exception as e:
                        doc_contents.append(f"Error processing document: {e}")

                # Combine the formatted documents into a single context
                context = "\n".join(doc_contents)
                for subclaim in entry["subclaims"]:
                    gold_answer = (
                        " ".join(entry["gld_ans"])
                        if isinstance(entry["gld_ans"], list)
                        else entry["gld_ans"]
                    )
                    annotation = self.verifier.annotate(
                        entry["query"], gold_answer, context, subclaim["subclaim"]
                    )
                    subclaim["annotations"]["gpt"] = annotation
        with open(self.subclaims_file, "w", encoding="utf-8") as jsonfile:
            json.dump(subclaims_data, jsonfile, indent=4)
        print(f"Subclaims with annotations saved to {self.subclaims_file}.")


def process_subclaims(
    query_path, subclaims_path, faiss_manager, scorer, truncation_strategy, truncate_by
):
    # Check if the file exists and load it if it does
    data = None
    if os.path.exists(subclaims_path):
        data = load_subclaim_data(subclaims_path)

        # Check if the data is valid
        score_method_to_check = [
            "relavance",
            "frequency",
            "cosine_similarity",
            "random",
            "ordinal",
        ]  # TODO
        if data:
            needs_scoring = any(
                len(subclaim["scores"]) == 0
                for pt in data
                for subclaim in pt["subclaims"]
            ) or any(
                score_method not in subclaim["scores"].keys()
                for pt in data
                for subclaim in pt["subclaims"]
                for score_method in score_method_to_check
            )

            needs_annotation = any(
                len(subclaim["annotations"]) == 0
                for pt in data
                for subclaim in pt["subclaims"]
            )

            if not (needs_scoring or needs_annotation):
                print(f"Subclaims data already exists in {subclaims_path}.")
                return data

    # Initialize processor only when needed
    processor = SubclaimProcessor(faiss_manager, scorer, subclaims_path)

    # Generate subclaims if data doesn't exist
    if not data:
        processor.get_subclaims(
            query_path, truncation_strategy=truncation_strategy, truncate_by=truncate_by
        )
        processor.score_subclaim()
        processor.annotate_subclaim()
    else:
        # Update only what's needed
        if needs_scoring:
            processor.score_subclaim()
        if needs_annotation:
            processor.annotate_subclaim()

    return load_subclaim_data(subclaims_path)


# Example usage
if __name__ == "__main__":
    document_file = FileManager(
        "data/processed/FactScore/sampled_10_fact_score_documents.txt"
    )
    # load the txt file into file manager
    faiss_manager = FAISSIndexManager()
    # load file manager text into faiss index
    faiss_manager.upsert_file_to_faiss(document_file)
    scorer = SubclaimScorer()
    processor = SubclaimProcessor(faiss_manager, scorer)
    processor.get_subclaims(
        "data/processed/FactScore/sampled_10_fact_score_queries.json",
        "data/out/FactScore/fact_score_10_subclaims.json",
    )
    processor.score_subclaim()
    processor.annotate_subclaim()
