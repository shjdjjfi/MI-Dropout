DIRECT_TEMPLATE = """Answer the question directly and briefly.\nQuestion: {question}\nContext:\n{context}\nAnswer:"""

COT_TEMPLATE = """Solve the multi-hop reasoning problem step by step.\nQuestion: {question}\nContext:\n{context}\nReasoning then answer:"""

CONCISE_COT_TEMPLATE = """Reason in <= {budget} tokens and output final answer.\nQuestion: {question}\nContext:\n{context}\nReasoning:"""

GRAPH_TEMPLATE = """Use the reasoning graph nodes and edges to answer.\nQuestion: {question}\nNodes:\n{nodes}\nEdges:\n{edges}\nFinal answer:"""
