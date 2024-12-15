FEW_SHOT_CONSTRUCT_GRAPH = """
### Task: Construct a graph that captures entities and relationships from a given claim. Extract triplets with entities and relations between them, including hidden, ambiguous or implicit entities.

### Guidelines:

1. Only use information from the claim; do not include external knowledge.
2. Do not repeat similar triplets in the graph.
3. Return the graph in the following format:
```
<guidance_for_graph_construction>
<graph>
<entity_1>||<relationship>||<entity_2>
```
* `<guidance_for_graph_construction>`: a detail explanation of how to construct the graph
* `<information>`: the information in the original claim used to extract the following triplets
**Entity and Relationship Format:**
* `<entity_1>`: source entity (use `hidden_entity_{index}` for implicit entities)
* `<entity_2>`: target entity (use `hidden_entity_{index}` for implicit entities)
* `<relationship>`: short descriptive text describing the relationship between entities

### Examples:

{{examples_text}}

### Input Claim:
{{claim}}

Return the constructed graph in the specified format.
""".strip()

FEW_SHOT_CONSTRUCT_GRAPH_EXAMPLE_TEMPLATE = """
-- Example --
<input_claim> {{claim}}
<guidance_for_graph_construction>
{{guidance_for_graph_construction}}
<graph>
{{graph}}
""".strip()