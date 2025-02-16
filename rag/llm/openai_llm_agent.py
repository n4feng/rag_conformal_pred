from openai import OpenAI
from llm.llm_agent import LLMAgent
import os

class OpenAILLMAgent(LLMAgent):
    def __init__(self, openai_key, instruction: str, model="gpt-4o-mini"):
        os.environ["OPENAI_API_KEY"] = openai_key
        self.instruction = instruction
        self.model = model
        self.client = OpenAI()

    def answer(self, question) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": question
                }
            ],
            timeout=600,
        )
        return completion.choices[0].message.content
    
    def preProcess(self, query):
        return query
    
    def postProcess(self, response):
        return response