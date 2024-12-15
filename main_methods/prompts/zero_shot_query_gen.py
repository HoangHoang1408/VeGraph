ZERO_SHOT_QUERY_GEN = """
### Task: You will be given a claim and a graph via triplet form. In the graph, there will have a entity that is hidden (hidden_entity) that needs to be resolved via searching in an external knowledge base. Your job is to try to generate the search question to resolve this hidden entity.

### Task Notes:
1) Graph Triplet will provided with triplets following the form: triplet_id||entity1||relationship||entity2.
2) The hidden entity need to be resolved will be marked as 'hidden_entity' in the graph.
3) You MUST generate one question to resolve the 'hidden_entity' in the graph.
4) You do NOT need to combine all the information of the triplets to from the question to resolve the entity. Try one or more aspects corresponding to triplets at a time that is enough to form the question to identify that entity.

### Return with the following JSON format and do NOT include other unnecessary details beyond the JSON object:
{
    "rationale": "a short rationale explaining how you use the information to generate the query",
    "question": "generated search question to resolve the entity",
    "triplet_ids": "a list containing ids of the triplet with information is used to generate the query"
}

### Actual input:
Claim: {{claim}}
Graph:
{{graph}}

### Now generate a question to resolve the hidden_entity and return in correct format.
""".strip()