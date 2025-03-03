import os
import ast
import bz2
import sqlite3
import json
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def load_qa_data(self, ouptut_path: str):
        if os.path.exists(ouptut_path):
            print(f"Dataset already exists at {ouptut_path}.")
        else:
            print(f"Loading {self.dataset} dataset.")    
            if self.dataset == "fact_score":
                self.dataset = load_fact_score_data(ouptut_path)
            elif self.dataset == "hotpot_qa":
                self.dataset = load_hotpot_qa_data(ouptut_path)
            elif self.dataset == "pop_qa":
                self.dataset = load_pop_qa_data(ouptut_path)

    def create_wiki_db(self, source_path: str ='data/raw/WikiDB/enwiki-20171001-pages-meta-current-withlinks-abstracts', output_path: str = 'data/raw/WikiDB/enwiki_20190113.db'):
        "Create a SQLite database from the Wikipedia dump data."

        if os.path.exists(output_path):
            print(f"Database already exists at {output_path}.")
            return
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path {source_path} not found.")
        else:
            print(f"Reading data from {source_path}")
            # Create a connection to the SQLite database
            conn = sqlite3.connect(output_path)
            cursor = conn.cursor()

            # Create a table to store the content
            cursor.execute('''DROP TABLE IF EXISTS wiki_content''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS wiki_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                url TEXT,
                text TEXT
            )
            ''')


            # Iterate through each bz2 file in the folder
            for folder in os.listdir(source_path):
                folder_path = f'{source_path}/{folder}'
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.bz2'):
                        file_path = os.path.join(folder_path, file_name)
                        with bz2.open(file_path, 'rt') as file:
                            content = file.read()
                            lines = content.split('\n')
                            for line in lines:
                                if line.strip():
                                    data = json.loads(line)
                                    line = ast.literal_eval(line)
                                    id = line.get('id', '')
                                    title = line['title']
                                    url = line.get('url', '')
                                    text = str(line.get('text', ''))
                                    cursor.execute('''
                                    INSERT INTO wiki_content (id, title, url, text)
                                    VALUES (?, ?, ?, ?)
                                    ''', (id, title, url, text))
                                    # print(f'Inserted {title} into the database')

            # Commit the changes and close the connection
            conn.commit()
            conn.close()
            print(f'Created database at {output_path}')


def load_fact_score_data():
    raise NotImplementedError

def load_hotpot_qa_data(output_path: str):
    """Load HotpotQA dataset and save validation set to json file."""

    dataset = load_dataset('kilt_tasks', 'hotpotqa')
    dataset['validation'].to_json(output_path, orient='records', lines=True)
    print("HotpotQA validation set saved to", output_path)
    
    return

def load_pop_qa_data(output_path: str = "data/raw/PopQA/popQA_test.json"):
    """Load PopQA dataset and save test set to json file."""

    dataset = load_dataset('akariasai/popQA')
    dataset['test'].to_json(output_path, orient='records', lines=True)
    print("PopQA test set saved to", output_path)

    return

# example code
if __name__ == "__main__":
    # loader = DataLoader("fact_score")
    # loader.load_qa_data("data/raw/FactScore/factscore_names.txt")
    
    # loader = DataLoader("hotpot_qa")
    # loader.load_qa_data("data/raw/HotpotQA/hotpotqa_validation_set.jsonl")
    
    loader = DataLoader("pop_qa")
    loader.load_qa_data("data/raw/PopQA/popQA_test.json")

    loader.create_wiki_db()


        