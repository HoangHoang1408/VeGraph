ZERO_SHOT_QUERY_REFINE = """
### Task: You will be given a claim and a graph via triplet form.
In the graph, there will have a entity that is hidden (hidden_entity) that needs to be resolved via searching in the external knowledge base.
You have already make some trials with thoughts and questions to try to get the hidden_entity, but failed.
Your job is to try to refine and regenerate a new search question to resolve this hidden entity.

### Task Notes:
1) Graph will provided triplets following the form: triplet_id||entity1||relationship||entity2.
2) Graph can contain triplets with information that is resolved via searching in the knowledge base compared to the claim.
3) The hidden entity need to be resolved will be marked as 'hidden_entity' in the graph.
4) You MUST generate one new question to resolve the 'hidden_entity' in the graph.
5) You do NOT need to combine all the information of the triplets to from the question to resolve the entity. Try one or more proper aspects corresponding to triplets at a time to form the question to identify hidden_entity.

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

### Failed rationales and questions:
{{failed_trials_text}}

### Now generate a new question to resolve the hidden_entity and return in correct format
""".strip()

ZERO_SHOT_QUERY_REFINE_FAILED_TRIAL_TEMPLATE = """
-- Failed trial {{i}} --
<old_rationale> {{rationale}}
<old_question> {{question}}
""".strip()