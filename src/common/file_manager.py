import os
import json
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FileManager:
    def __init__(self, file_path, chunk_size=2000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.texts = []
        directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.texts_file = os.path.join(directory, f"{base_name}_texts.json")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=200)

        # Load texts from file if it exists
        if os.path.exists(self.texts_file):
            with open(self.texts_file, "r") as f:
                self.texts = json.load(f)
            print(f"Loaded texts from file: {self.texts_file}")

    def load_pdf_document(self):
        pdf_reader = PdfReader(self.file_path)
        documents = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:  # Ensure the page has text
                document = Document(
                    metadata={'source': self.file_path, 'page': page_num},
                    page_content=page_text
                )
                documents.append(document)

        return documents

    def dump_documents(self, documents):
        self.texts = [(i, str(doc)) for i, doc in enumerate(documents)]
        if self.texts:
            with open(self.texts_file, "w") as f:
                json.dump(self.texts, f)
            print(f"Associated texts saved to file: {self.texts_file}")

    def process_pdf(self):
        data = self.load_pdf_document()
        
        documents = self.text_splitter.split_documents(data)
        self.dump_documents(documents)

        return self.texts
    
    def process_wiki_embedding(self):
        if not self.texts:
            with open(self.file_path, 'r', encoding='utf-8') as f:
            # Load the file content as a dictionary
                data = json.load(f)
                documents = []
                for title, texts in data.items():
                    # Create embeddings for each text
                    for text in texts:
                        doc = Document(page_content=title+": "+text, metadata={'source': title, 'file_path': self.file_path})
                        documents.append(doc)
                self.dump_documents(documents)
    