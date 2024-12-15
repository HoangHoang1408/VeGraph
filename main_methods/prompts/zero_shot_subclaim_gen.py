# ZERO_SHOT_SUBCLAIM_GEN = """
# ### Task: You will be given a claim and a graph via triplets form.
# Each triplet is a subclaim that need to verify its veracity via searching through the external knowledge base.
# Your job is to try to generate a full subclaim (with each subclaim is derived from a triplet) to verify those subclaims.

# ### Task Notes:
# 1) For each triplet in the graph, you need to generate one subclaim that can be verify with yes or no corresponding to the triplet's veracity.
# 2) You can use additional information from the claim to generate proper subclaims.

# ### Return with the following JSON format and do NOT include other unnecessary details beyond the JSON object:
# {"subclaims": [a list of subclaims]}

# ### Claim: {{claim}}

# ### Graph:
# {{graph}}

# ### Now generate the subclaim list and return in correct format.
# """.strip()

ZERO_SHOT_SUBCLAIM_GEN = """
### Task:
You will be given a claim and a graph represented as triplets. The graph is divided into two parts: verified triplets and unverified triplets. Each triplet represents a subclaim from the original claim.
For each triplet in the unverified triplets part, convert this triplet to a text claim to verify. Do NOT generate subclaims for verified triplets.
You can use additional information from the claim to generate proper text claim.

### Output format: Return in the following JSON format and do NOT include unnecessary details beyond the JSON object:
{"subclaims": [list of generated subclaims to verify]}

### Claim: {{claim}}
### Verified Triplets
{{verified_graph}}

### Unverified Triplets
{{unverified_graph}}

### Now generate the subclaim list and return in correct format.
""".strip()
