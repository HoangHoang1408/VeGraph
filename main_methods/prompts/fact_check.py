# FACT_CHECK_WITH_DOCS = """
# ### Task: Verify the following claim based on the information in the provided documents.

# ### Guidelines:
# 1) Use only the content from the provided documents to verify the claim. Do **not** rely on outside information or generate knowledge yourself. Avoid making implications.
# 2) To verify claim:
#     - Return **true** if the claim is supported by the documents.
#     - Return **false** if the documents provide information that contradicts the claim.
# 3) Return in the following JSON format:
# {
#     "rationale": "a short rationale with evidence or contradictions to guide the verifying process",
#     "veracity": true or false
# }
# 4) If the claim cannot be answer due to insufficient information from provided documents, return:
# {
#     "rationale": null,
#     "veracity": null
# }

# ### Documents:
# {{context}}

# ### Question:
# {{claim}}
# """.strip()

FACT_CHECK_WITH_DOCS = """
### Task: Verify the following claim based on the information in the provided documents.

### Guidelines:
1) Use only the content from the provided documents to verify the claim. Do NOT rely on outside information or generate knowledge yourself. Avoid making implications.
2) To verify claim:
    - Return true if the claim is supported by the documents.
    - Return false if the documents provide information that contradicts the claim.
3) Return in the following JSON format:
{
    "rationale": "A short rationale with supported or contradicted evidence to guide the verifying process. You should explicitly mention the evidence and do NOT mention the provided documents in this rationale.",
    "veracity": true or false
}
4) If the claim cannot be answer due to insufficient information from provided documents, return:
{
    "rationale": null,
    "veracity": null
}

### Documents:
{{context}}

### Claim:
{{claim}}
""".strip()