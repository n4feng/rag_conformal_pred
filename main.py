import os
import argparse

from src.data_processor.dataloader import DataLoader
from src.data_processor.query_processor import QueryProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset", choices=["fact_score", "hotpot_qa", "pop_qa"])
    parser.add_argument("--wiki_db_dir", type=str, default="enwiki-20230401.db", help="Name of the dataset", choices=["fact_score", "hotpot_qa", "pop_qa"])
    args = parser.parse_args()

    # Set raw data directory
    if args.dataset == "fact_score":
        raw_data_dir = "data/raw/FactScore"
        processed_data_dir = "data/processed/FactScore"
    elif args.dataset == "hotpot_qa": 
        raw_data_dir = "data/raw/HotpotQA"
        processed_data_dir = "data/processed/HotpotQA"
    elif args.dataset == "pop_qa":  
        raw_data_dir = "data/raw/PopQA"
        processed_data_dir = "data/processed/PopQA"
    
    # Load data
    if not os.path.exists(f"{raw_data_dir}/raw_{args.dataset}.json"):
        data_loader = DataLoader(args.dataset)
        data_loader.load_qa_data(ouptut_path=f"{raw_data_dir}/raw_{args.dataset}.json")
        if not os.path.exists(f"data/raw/WikiDB/{args.wiki_db_dir}"):
            data_loader.create_wiki_db(source_path='data/raw/WikiDB/enwiki-20171001-pages-meta-current-withlinks-abstracts', output_path='data/raw/WikiDB/enwiki_20190113.db')

    # Process data
    if not os.path.isfile(f"data/raw/WikiDB/{args.wiki_db_dir}"):
        raise FileNotFoundError(f"Database file 'data/raw/WikiDB/{args.wiki_db_dir}' not found.")
    else:
        query_processor = QueryProcessor(db_path=f"data/raw/WikiDB/{args.wiki_db_dir}")
        # create queries data
        query_processor.get_queries(dataset=args.dataset, input_file=f"{raw_data_dir}/raw_{args.dataset}.json", output_file=f"{processed_data_dir}/{args.dataset}_queries.json")
        # create documents data
        query_processor.get_documents(query_file=f"{processed_data_dir}/{args.dataset}_queries.json", output_file=f"{processed_data_dir}/{args.dataset}_documents.txt")