import json
from tqdm import tqdm
from jsonschema import RefResolver, validate
from src.subclaim_processor.query_processor import IQueryProcessor
from src.common.llm.openai_rag_agent import OpenAIRAGAgent
from src.common.llm.openai_atomicfact_generator import OpenAIAtomicFactGenerator
from src.common.llm.openai_claim_verification import OpenAIClaimVerification
from src.subclaim_processor.scorer.base_scorer import IScorer
from src.subclaim_processor.scorer.document_scorer import IDocumentScorer
from src.subclaim_processor.scorer.similarity_scorer import SimilarityScorer
from src.common.faiss_manager import FAISSIndexManager
from src.common.file_manager import FileManager


class SubclaimProcessor(IQueryProcessor):
    def __init__(
        self,
        faiss_manager,
        scorer: IScorer,
    ):
        self.faiss_manager = faiss_manager
        self.response_agent = OpenAIRAGAgent(faiss_manager)
        self.generator = OpenAIAtomicFactGenerator()
        self.verifier = OpenAIClaimVerification()
        self.scorer = scorer
        with open(
            "data/out/subclaims_schema.json", "r", encoding="utf-8"
        ) as schemafile:
            self.subclaim_schema = json.load(schemafile)

    def get_subclaims(self, query_file: str, subclaims_file: str):
        self.subclaims_file = subclaims_file
        with open(query_file, "r", encoding="utf-8") as jsonfile:
            queries = json.load(jsonfile)

        with open(subclaims_file, "w", encoding="utf-8") as subclaimsfile:
            subclaimsfile.write("[\n")

        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            question = query["input"]

            # document retrieval
            self.retrieved_docs = retrieved_docs = (
                self.faiss_manager.search_faiss_index(question, top_k=10, threshold=0.3)
            )

            response = self.response_agent.answer(question, retrieved_docs)
            subclaims = self.generator.get_facts_from_text(response)
            subclaims_entry = {
                "query": question,
                "gld_ans": query["output"]["answer"],
                "retrieved_docs": retrieved_docs,
                "response": response,
                "subclaims": [],
            }

            for subclaim in subclaims:
                subclaims_entry["subclaims"].append(
                    {
                        "subclaim": subclaim,
                        "scores": {},  # Add logic to populate scores if available
                        "annotations": {},  # Add logic to populate annotations if available
                    }
                )

            with open(subclaims_file, "a", encoding="utf-8") as subclaimsfile:
                json.dump(subclaims_entry, subclaimsfile, indent=4)
                if i < len(queries) - 1:
                    subclaimsfile.write(",\n")

        with open(subclaims_file, "a", encoding="utf-8") as subclaimsfile:
            subclaimsfile.write("\n]")
        print(f"Query responses saved to {subclaims_file}.")

    def score_subclaim(self):
        with open(self.subclaims_file, "r", encoding="utf-8") as jsonfile:
            subclaims_data = json.load(jsonfile)
            for entry in tqdm(subclaims_data, desc="Scoring subclaims"):
                validate(instance=entry, schema=self.subclaim_schema)
                for subclaim in entry["subclaims"]:
                    score = 0.0
                    if isinstance(self.scorer, IDocumentScorer):
                        score = self.scorer.score(
                            subclaim["subclaim"], self.retrieved_docs
                        )
                    else:
                        score = self.scorer.score(subclaim["subclaim"])
                    subclaim["scores"][self.scorer.get_type()] = float(score)

        with open(self.subclaims_file, "w", encoding="utf-8") as jsonfile:
            json.dump(subclaims_data, jsonfile, indent=4)
        print(f"Subclaims with scores saved to {self.subclaims_file}.")

    def annotate_subclaim(self):
        with open(self.subclaims_file, "r", encoding="utf-8") as jsonfile:
            subclaims_data = json.load(jsonfile)
            for entry in tqdm(subclaims_data, desc="Annotating subclaims"):
                validate(instance=entry, schema=self.subclaim_schema)
                doc_contents = []
                for doc in self.retrieved_docs:
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
                    annotation = self.verifier.annotate(
                        entry["query"], entry["gld_ans"], context, subclaim["subclaim"]
                    )
                    subclaim["annotations"]["gpt"] = annotation
        with open(self.subclaims_file, "w", encoding="utf-8") as jsonfile:
            json.dump(subclaims_data, jsonfile, indent=4)
        print(f"Subclaims with annotations saved to {self.subclaims_file}.")


# Example usage
if __name__ == "__main__":
    document_file = FileManager(
        "data/processed/FactScore/sampled_10_fact_score_documents.txt"
    )
    # load the txt file into file manager
    faiss_manager = FAISSIndexManager()
    # load file manager text into faiss index
    faiss_manager.upsert_file_to_faiss(document_file)
    scorer = SimilarityScorer()
    processor = SubclaimProcessor(faiss_manager, scorer)
    processor.get_subclaims(
        "data/processed/FactScore/sampled_10_fact_score_queries.json",
        "data/out/FactScore/fact_score_10_subclaims.json",
    )
    processor.score_subclaim()
    processor.annotate_subclaim()
