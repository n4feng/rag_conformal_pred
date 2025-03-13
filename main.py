import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.dataloader.dataloader import DataLoader
from src.data_processor.query_processor import QueryProcessor
from src.subclaim_processor.scorer.similarity_scorer import SimilarityScorer

from src.calibration.conformal import ConformalCalibration
from src.calibration.utils import load_calibration

from src.common.llm.openai_atomicfact_generator import OpenAIAtomicFactGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset",
        choices=["fact_score", "hotpot_qa", "pop_qa", "medlf_qa"],
    )
    parser.add_argument(
        "--wiki_db_dir",
        type=str,
        default="enwiki-20230401.db",
        help="Name of the Wiki database file",
    )
    parser.add_argument(
        "--query_size", type=int, default=10, help="Number of queries to sample"
    )
    parser.add_argument(
        "--delete_existing_index", action="store_true", help="Delete Faiss index"
    )
    # parser.add_argument(
    #     "--embedding_model",
    #     type=str,
    #     default="text-embedding-3-large",
    #     help="Name of the embedding model for Faiss index",
    # )
    # parser.add_argument(
    #     "--retrieve_top_k", type=int, default=10, help="Number of documents to retrieve"
    # )
    # parser.add_argument(
    #     "--run_standard_conformal_prediction",
    #     action="store_true",
    #     help="Run full conformal prediction",
    # )
    # parser.add_argument(
    #     "--run_group_conditional_conformal",
    #     action="store_true",
    #     help="Run group conditional conformal prediction",
    # )
    # parser.add_argument("--a", type=float, default=1.0)
    # parser.add_argument("--confidence_methods", type=str, default="similarity")
    args = parser.parse_args()

    ####################################### Data and Folder Set up ############################################
    # Define dataset mappings with associated index store
    conformal_alphas = np.arange(0.05, 0.45, 0.05)

    DATASET_CONFIG = {
        "fact_score": {"name": "FactScore", "index_store": "index_store/Wiki"},
        "hotpot_qa": {"name": "HotpotQA", "index_store": "index_store/Wiki"},
        "pop_qa": {"name": "PopQA", "index_store": "index_store/Wiki"},
        "medlf_qa": {"name": "MedLFQA", "index_store": "index_store/MedLFQA"},
    }

    # Get dataset configuration or fail early
    dataset_config = DATASET_CONFIG.get(args.dataset)
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_name = dataset_config["name"]
    index_store_dir = dataset_config["index_store"]

    # Set data directories using os.path.join for cross-platform compatibility
    raw_data_dir = os.path.join("data", "raw", dataset_name)
    processed_data_dir = os.path.join("data", "processed", dataset_name)
    response_dir = os.path.join("data", "out", dataset_name)
    wiki_data_dir = os.path.join("data", "raw", "WikiDB", args.wiki_db_dir)

    # Check if wiki database exists early # TODO remove this check if dataset is not Wiki - need to fix DocDB class accordingly
    if not os.path.isfile(wiki_data_dir):
        raise FileNotFoundError(f"Database file '{wiki_data_dir}' not found.")

    # Determine raw data file path
    if args.dataset == "medlfqa":
        input_file = os.path.join("data", "raw", "MedLFQA")
        raw_data_path = input_file
    else:
        raw_data_file = f"raw_{args.dataset}.json"
        raw_data_path = os.path.join(raw_data_dir, raw_data_file)

    # Load data if needed
    if not os.path.exists(raw_data_path):
        # Create directories if they don't exist
        os.makedirs(raw_data_dir, exist_ok=True)

        data_loader = DataLoader(args.dataset)
        data_loader.load_qa_data(ouptut_path=raw_data_path)

        wiki_db_path = os.path.join("data", "raw", "WikiDB", args.wiki_db_dir)
        if not os.path.exists(wiki_db_path):
            wiki_source = (
                "data/raw/WikiDB/enwiki-20171001-pages-meta-current-withlinks-abstracts"
            )
            wiki_output = "data/raw/WikiDB/enwiki_20190113.db"
            data_loader.create_wiki_db(source_path=wiki_source, output_path=wiki_output)

    # Process queries and documents
    if args.dataset == "medlf_qa":
        input_file = os.path.join("data", "raw", "MedLFQA")
    else:
        input_file = raw_data_path
    query_output_file = f"{args.dataset}_queries.json"
    document_output_file = f"{args.dataset}_documents.txt"

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_response_path = os.path.join(
        response_dir, f"{args.dataset}_{args.query_size}_raw_response.json"
    )
    subclaims_path = os.path.join(
        response_dir, f"{args.dataset}_{args.query_size}_subclaims_with_scores.json"
    )
    result_fig_path = f"data/result/{dataset_name}/{args.dataset}_{args.confidence_methods}_a={args.a:.2f}_removal_fig.png"
    result_path = f"data/result/{dataset_name}/{args.dataset}_{args.confidence_methods}_a={args.a:.2f}_removal.csv"
    ####################################### End of Data and Folder Set up ######################################

    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    query_processor = QueryProcessor(db_path=wiki_data_dir, query_size=args.query_size)

    # Create queries data
    queries, query_path = query_processor.get_queries(
        dataset=args.dataset,
        input_file=input_file,
        output_dir=processed_data_dir,
        output_file=query_output_file,
    )
    print(f"Query size: {len(queries)}")

    # Create documents data
    document_path = query_processor.get_documents(
        query_dir=query_path,
        output_dir=processed_data_dir,
        output_file=document_output_file,
    )

    # # Index creation and retrieval
    # # TODO: break apart embedding and scorer
    # wikitexts_embedding = SimilarityScorer(
    #     embedding_model=args.embedding_model,
    #     index_path=f"{index_store_dir}/index.faiss",
    #     indice2fm_path=f"{index_store_dir}/indice2fm.json",
    # )

    # if args.delete_existing_index:
    #     wikitexts_embedding.faiss_manager.delete_faiss_index()
    # # create index if it does not exist

    # if not os.path.exists(f"{index_store_dir}/index.faiss"):
    #     wikitexts_embedding.create_or_update_index(document_path)
    # elif (
    #     document_path not in wikitexts_embedding.faiss_manager.indice2fm.keys()
    #     and wikitexts_embedding.faiss_manager.is_indice_align()
    # ):
    #     print(
    #         f"There has exist an index, However, document '{document_path}' is not indexed. Adding it to the index store."
    #     )
    #     wikitexts_embedding.create_or_update_index(document_path)

    # # check if index and indice2fm are aligned
    # if not wikitexts_embedding.faiss_manager.is_indice_align():
    #     raise ValueError("Index and indice2fm are not aligned.")

    # # query the index and generate response
    # responses = {}
    # retrieved_docs_all = {}
    # for query in tqdm(queries.keys(), desc="Processing queries"):
    #     retrieved_docs = wikitexts_embedding.faiss_manager.search_faiss_index(
    #         query, top_k=10
    #     )
    #     retrieved_docs_all[query] = retrieved_docs
    #     response = wikitexts_embedding.faiss_manager.generate_response_from_context(
    #         query, retrieved_docs
    #     )
    #     responses[query] = response

    # # set up results directory
    # if not os.path.exists(response_dir):
    #     os.makedirs(response_dir)

    # # baseline scoring of the responses without removing subclaims
    # for query, docs in retrieved_docs_all.items():
    #     similarity_score = wikitexts_embedding.score(query, docs)

    #     # Load existing responses if the file exists
    #     existing_responses = {}
    #     if os.path.exists(raw_response_path):
    #         with open(raw_response_path, "r", encoding="utf-8") as f:
    #             for line in f:
    #                 record = json.loads(line)
    #                 existing_responses[record["query"]] = record

    #     # Append new responses only if they are not already in the file
    #     with open(raw_response_path, "a", encoding="utf-8") as f:
    #         for query, response in responses.items():
    #             if query not in existing_responses:
    #                 f.write(
    #                     json.dumps(
    #                         {
    #                             "query": query,
    #                             "response": response,
    #                             "similarity_score": f"{similarity_score:.2f}",
    #                         }
    #                     )
    #                     + "\n"
    #                 )

    # print(f"Raw responses saved to '{raw_response_path}'")

    # # generate subclaims from the response
    # if not os.path.exists(subclaims_path):
    #     print(f"Generating subclaims at '{subclaims_path}'")
    #     with open(subclaims_path, "w", encoding="utf-8") as f:
    #         f.write("")
    #     # generate subclaims if not exist
    #     gen = OpenAIAtomicFactGenerator()
    #     for query, response in responses.items():
    #         atomicFacts = gen.get_facts_from_text(response)
    #         subclaims_score = {}
    #         for fact in atomicFacts:
    #             purefact = fact.rpartition(":")[0] if ":" in fact else fact
    #             score = wikitexts_embedding.score(purefact, retrieved_docs)
    #             subclaims_score[purefact] = float(score)

    #         subclaims_score = sorted(
    #             subclaims_score.items(), key=lambda x: x[1], reverse=True
    #         )
    #         ground_truth_answer = queries[query]
    #         raw_score = float(
    #             wikitexts_embedding.score(
    #                 query + " " + ground_truth_answer, retrieved_docs
    #             )
    #         )
    #         with open(subclaims_path, "a", encoding="utf-8") as f:
    #             f.write(
    #                 json.dumps(
    #                     {
    #                         "query": query,
    #                         "answer": ground_truth_answer,
    #                         "calibrate_score": f"{raw_score:.2f}",
    #                         "response": response,
    #                         "subclaims_score": subclaims_score,
    #                     }
    #                 )
    #                 + "\n"
    #             )

    #     print(f"Subclaims saved to '{subclaims_path}'")
    # else:
    #     print(f"Subclaims already exist at '{subclaims_path}'")

    # # calibration and conformal prediction results
    # data = load_calibration(subclaims_path)

    # if args.run_standard_conformal_prediction:
    #     conformal = ConformalCalibration(
    #         dataset_name=args.dataset, confidence_method=args.confidence_methods
    #     )
    # elif args.run_group_conditional_conformal:
    #     raise NotImplementedError("Group conditional conformal not implemented")
    # else:
    #     raise ValueError("Invalid calibration method")

    # conformal.plot_conformal_removal(
    #     data,
    #     conformal_alphas,
    #     args.a,
    #     result_fig_path,
    #     result_path,
    #     plot_group_results=False,
    # )
