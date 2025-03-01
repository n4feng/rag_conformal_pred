from openai import OpenAI
import os
from dotenv import load_dotenv
from src.common.llm.llm_agent import LLMAgent
from src.common.faiss_manager import FAISSIndexManager

class OpenAIRAGAgent(LLMAgent):
    def __init__(self, faiss_manager, instruction: str = "You are a helpful assistant that answers questions based on provided context.", model="gpt-4o-mini"):
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.instruction = instruction
        self.model = model
        self.client = OpenAI()
        self.faiss_manager = FAISSIndexManager()
        self.faiss_manager = faiss_manager

    def answer(self, question) -> str:
        retrieved_docs = self.faiss_manager.search_faiss_index(question, top_k=10, threshold=0.1)
        if not retrieved_docs:
            return "No relevant documents found in the FAISS index."

        # Process retrieved documents into a clean context
        formatted_docs = []
        for doc in retrieved_docs:
            try:
                # Split the document string into page_content and metadata
                doc_parts = doc.split("metadata=")
                page_content = doc_parts[0].replace("page_content=", "").strip()
                metadata = doc_parts[1].strip() if len(doc_parts) > 1 else "Unknown source"

                # Format each document clearly
                formatted_doc = f"Content: {page_content}\nSource: {metadata}"
                formatted_docs.append(formatted_doc)
            except Exception as e:
                formatted_docs.append(f"Error processing document: {e}")

        # Combine the formatted documents into a single context
        context = "\n\n---\n\n".join(formatted_docs)

        # Construct the prompt for the OpenAI API
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"The following context was retrieved from the database:\n\n{context}"}
        ]

        # Generate response using OpenAI Chat API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    
    def preProcess(self, query):
        return query
    
    def postProcess(self, response):
        return response