import os
import argparse

from src.dataloader.dataloader import DataLoader
from src.data_processor.query_processor import QueryProcessor

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
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-large",
        help="Name of the embedding model for Faiss index",
    )
    parser.add_argument(
        "--retrieve_top_k", type=int, default=10, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--full_conformal", action="store_true", help="Run full conformal prediction"
    )
    parser.add_argument(
        "--group_conditional_conformal",
        action="store_true",
        help="Run group conditional conformal prediction",
    )
    parser.add_argument(
        "--calibration_score_path",
        type=str,
        default="data/out/calibration_score.json",
        help="Path to save calibration scores",
    )
    args = parser.parse_args()

    # Define dataset mappings with associated index store
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
    print(f"Document size: {len(document_path)}")
