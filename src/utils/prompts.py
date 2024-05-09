# ------------------------------------------------------------------------
# Dec 2023: identify checkworthy
# ------------------------------------------------------------------------
CHECKWORTHY_PROMPT = """
Your task is to identify whether texts are checkworthy in the context of fact-checking.
Let's define a function named checkworthy(input: List[str]).
The return value should be a list of strings, where each string selects from ["Yes", "No"].
"Yes" means the text is a factual checkworthy statement.
"No" means that the text is not checkworthy, it might be an opinion, a question, or others.
For example, if a user call checkworthy(["I think Apple is a good company.", "Friends is a great TV series.", "Are you sure Preslav is a professor in MBZUAI?", "The Stanford Prison Experiment was conducted in the basement of Encina Hall.", "As a language model, I can't provide these info."])
You should return a python list without any other words, 
["No", "Yes", "No", "Yes", "No"], with the same order and length as the input list.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

checkworthy({texts})
"""

SPECIFY_CHECKWORTHY_CATEGORY_PROMPT = """
You are a factchecker assistant with task to identify a sentence, whether it is 1. a factual claim; 2. an opinion; 3. not a claim (like a question or a imperative sentence); 4. other categories.
Let's define a function named checkworthy(input: str).
The return value should be a python int without any other words, representing index label, where index selects from [1, 2, 3, 4].

For example, if a user call checkworthy("I think Apple is a good company.")
You should return 2
If a user call checkworthy("Friends is a great TV series.")
You should return 1
If a user call checkworthy("Are you sure Preslav is a professor in MBZUAI?")
You should return 3
If a user call checkworthy("As a language model, I can't provide these info.")
You should return 4
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

checkworthy("{sentence}")
"""
