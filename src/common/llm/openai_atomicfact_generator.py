import os
from dotenv import load_dotenv
from openai import OpenAI
from src.common import string_utils


class OpenAIAtomicFactGenerator(object):
    def __init__(self):
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.instruction = """Break down the following paragraph into individual, factual statements, 
                using clear and specific subjects (e.g., 'The company,' 'The report,' etc.), avoiding pronouns. 
                Evaluate each statement based on the paragraph's content and categorize it as one of the following:
                Supported - If the paragraph explicitly confirms the fact.
                Ambiguous - If the fact is suggested but lacks sufficient clarity.
                Not Supported - If the paragraph provides no evidence for the fact.
                Output the results as a single array of pairs in the format [${fact1}:supported; ${fact2}:ambiguous; ${fact3}:not supported; ...], 
                with each pair following this pattern. Do not include new lines. Make sure delimeter is always ; sign.
                Paragraph to analyze: """
        self.client = OpenAI()

    def get_atomic_facts_from_paragraph(self, paragraph):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant to breakdown long knowledge intensive text into independent fact"},
                {
                    "role": "user",
                    "content": self.instruction+paragraph
                }
            ]
        )
        return completion.choices[0].message.content

    def get_contents_from_title(self, db, title):
        data_list = db.get_text_from_title(title)
        contents = ""
        for data in data_list:
            contents += (data["text"])
        return contents
        
    def get_facts_from_title(self, db, title):
        contents = self.get_contents_from_title(db=db, title=title)
        facts = []
        #reason to do this is because the text in db sometimes are break in middle of a whole content
        #so re-joined all items in data, then re-break by <s></s> tag
        paragraphs = string_utils.extract_tag_content(contents)
    
        for paragraph in paragraphs:
            response = self.get_atomic_facts_from_paragraph(paragraph)
            result = string_utils.extract_array_result(response)
            facts.extend(string_utils.extract_string_array(result))
            #print("array list get extracted from is: " + result)
        return facts

    def get_facts_from_text(self, text):
        response = self.get_atomic_facts_from_paragraph(text)
        result = string_utils.extract_array_result(response)
        facts = (string_utils.extract_string_array(result))
        #print("array list get extracted from is: " + result)

        return facts