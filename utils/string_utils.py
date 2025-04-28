import re

def extract_tag_content(input_string):
    # Use regular expression to find all content between <s> and </s> tags
    pattern = re.compile(r'<s>(.*?)</s>')
    matches = pattern.findall(input_string)
    return matches
    
def extract_array_result(input_string):
    # Use regular expression to find array result betweenn []
    # Assume input only have 1 array, or just want to get first array
    pattern = re.compile(r'\[(.*?)\]', re.DOTALL) # Use DOTALL to handle multiline content
    matches = pattern.findall(input_string)
    if len(matches) > 0:
        return matches[0]
    return "[]"
    
def extract_string_array(input_string):
    items = input_string.strip('[').strip(']').strip('"').strip().split(";")
    #remove empty string in list and trim \n
    items = [i.replace("\n", "") for i in items if i]
    return items

def categorize_question(question: str) -> str:
    """
    Categorize an English question into one of:
    - Yes/No
    - Why
    - Wh-
    - How
    - Others
    """

    # Normalize question
    q = question.strip().lower()
    
    # Patterns for different categories
    yes_no_starters = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "have", "has", "had", "should", "shall", "may", "might", "must")
    wh_words = ("what", "when", "where", "which", "who", "whom", "whose", "why")
    how_pattern = re.compile(r"^how(\b|\s)")

    # Categorization logic
    if q.startswith(yes_no_starters):
        return "Yes/No"
    elif q.startswith(wh_words):
        return "Wh-"
    elif how_pattern.match(q):
        return "How"
    elif " or " in q:
        return "Choice"
    else:
        return "Others"
