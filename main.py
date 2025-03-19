import os
import argparse
import numpy as np

from src.dataloader.dataloader import DataLoader
from src.data_processor.query_processor import QueryProcessor
from src.common.file_manager import FileManager
from src.common.faiss_manager import FAISSIndexManager
from src.subclaim_processor.scorer.similarity_scorer import SimilarityScorer
from src.subclaim_processor.subclaim_processor import process_subclaims
from src.calibration.conformal import SplitConformalCalibration


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
        "--wiki_db_file",
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
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-large",
        help="Name of the embedding model for Faiss index",
    )
    parser.add_argument(
        "--run_split_conformal_prediction",
        action="store_true",
        help="Run full conformal prediction",
    )
    parser.add_argument(
        "--run_group_conditional_conformal",
        action="store_true",
        help="Run group conditional conformal prediction",
    )
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--confidence_methods", type=str, default="similarity")
    args = parser.parse_args()

    ####################################### Data and Folder Set up ############################################
    # Define dataset mappings with associated index store
    conformal_alphas = np.arange(0.05, 0.45, 0.05)

    DATASET_CONFIG = {
        "fact_score": {"name": "FactScore", "index_store": "index_store/FactScore"},
        "hotpot_qa": {"name": "HotpotQA", "index_store": "index_store/HotpotQA"},
        "pop_qa": {"name": "PopQA", "index_store": "index_store/PopQA"},
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
    wiki_db_path = os.path.join("data", "raw", "WikiDB", args.wiki_db_file)

    # set up results directory
    for dir in [raw_data_dir, processed_data_dir, response_dir]:
        os.makedirs(dir, exist_ok=True)

    # Determine raw data file path
    if args.dataset == "medlfqa":
        input_file = os.path.join("data", "raw", "MedLFQA")
        raw_data_path = input_file
    else:
        raw_data_file = f"raw_{args.dataset}.json"
        raw_data_path = os.path.join(raw_data_dir, raw_data_file)

    # Load data if needed
    if not os.path.exists(raw_data_path):
        data_loader = DataLoader(args.dataset)
        data_loader.load_qa_data(ouptut_path=raw_data_path)

    # create wiki db if needed
    if not os.path.exists(wiki_db_path) or not os.path.isfile(wiki_db_path):
        wiki_source = (
            "data/raw/WikiDB/enwiki-20171001-pages-meta-current-withlinks-abstracts"
        )
        if not os.path.exists(wiki_source):
            raise FileNotFoundError(f"Wiki source data not found at {wiki_source}")
        data_loader.create_wiki_db(source_path=wiki_source, output_path=wiki_db_path)

    # Process queries and documents
    input_file = (
        os.path.join("data", "raw", "MedLFQA")
        if args.dataset == "medlf_qa"
        else raw_data_path
    )
    query_output_file = f"{args.dataset}_queries.json"
    document_output_file = f"{args.dataset}_documents.txt"

    subclaims_path = os.path.join(
        response_dir, f"{args.dataset}_{args.query_size}_subclaims_with_scores.json"
    )
    CP_result_fig_path = f"data/result/{dataset_name}/{args.dataset}_{args.confidence_methods}_a={args.a:.2f}_CP_removal.png"
    factual_result_fig_path = f"data/result/{dataset_name}/{args.dataset}_{args.confidence_methods}_a={args.a:.2f}_factual_removal.png"
    result_path = f"data/result/{dataset_name}/{args.dataset}_{args.confidence_methods}_a={args.a:.2f}_removal.csv"
    ####################################### End of Data and Folder Set up ######################################

    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    query_processor = QueryProcessor(db_path=wiki_db_path, query_size=args.query_size)

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

    # Index creation and retrieval
    index_file_path = os.path.join(index_store_dir, f"index_{args.query_size}.faiss")
    indice2fm_path = os.path.join(index_store_dir, f"indice2fm_{args.query_size}.json")
    faiss_manager = FAISSIndexManager(
        index_path=index_file_path, indice2fm_path=indice2fm_path
    )
    if args.delete_existing_index:
        faiss_manager.delete_index()

    # Create index if it does not exist
    print(document_path)
    document_file = FileManager(document_path)

    # If Index doesn't exist yet
    if not os.path.exists(index_file_path):
        try:
            faiss_manager.upsert_file_to_faiss(document_file)
            print(f"Created new index with document '{document_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to create new index: {str(e)}")

    # If Index exists but current document isn't indexed
    elif document_path not in faiss_manager.indice2fm:
        # Verify index integrity
        if not faiss_manager.is_indice_align():
            raise ValueError(
                "Index corruption detected: index and indice2fm are not aligned"
            )

        try:
            print(f"Adding document '{document_path}' to existing index")
            faiss_manager.upsert_file_to_faiss(document_file)
        except Exception as e:
            raise RuntimeError(f"Failed to add document to index: {str(e)}")

    # Case 3: Document is already indexed
    else:
        print(f"Document '{document_path}' is already indexed")

    # generate subclaims with scores
    scorer = SimilarityScorer(
        embedding_model=args.embedding_model,
        index_path=index_file_path,
        indice2fm_path=indice2fm_path,
    )

    subclaim_with_annotation_data = process_subclaims(
        query_path, subclaims_path, faiss_manager, scorer
    )

    # calibration and conformal prediction results
    if args.run_split_conformal_prediction:
        # TODO rename class
        conformal = SplitConformalCalibration(
            dataset_name=args.dataset, confidence_method=args.confidence_methods
        )
        conformal.plot_conformal_removal(
            data=subclaim_with_annotation_data,
            alphas=conformal_alphas,
            a=args.a,
            fig_filename=CP_result_fig_path,
            csv_filename=result_path,
        )
        conformal.plot_factual_removal(
            data=subclaim_with_annotation_data,
            alphas=conformal_alphas,
            a=args.a,
            fig_filename=factual_result_fig_path,
            csv_filename=result_path,
        )

    elif args.run_group_conditional_conformal:
        raise NotImplementedError("Group conditional conformal not implemented")
    else:
        raise ValueError("Invalid calibration method")
