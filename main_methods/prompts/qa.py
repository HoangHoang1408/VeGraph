QA_PROMPT_WITH_DOCS = """
### Task: Based only on the information provided in the given documents, answer the following question.

### Guidelines:
1) Your response must exclusively use information from the provided documents. Do **not** rely on outside knowledge or generate knowledge yourself.
2) Return only one the specific entity requested in the question formatted as follows:
{"answer": "the one entity you identitfied"}
3) If the entity is not found in the documents, return:
{"answer": null}

### Documents:
{{context}}

### Question:
{{question}}

### Now answer the question and return in correct format.
""".strip()