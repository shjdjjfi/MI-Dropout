from src.bottleneck.structured_ib import IBConfig, apply_structured_ib, extract_subgraph
from src.graph.induction import induce_graph


def test_graph_and_bottleneck_roundtrip() -> None:
    instance = {
        "id": "x1",
        "question": "Who wrote Hamlet?",
        "candidate_context": [{"title": "Hamlet", "text": "Hamlet is a tragedy written by William Shakespeare."}],
        "gold_answer": "William Shakespeare",
    }
    graph = induce_graph(instance)
    assert graph.number_of_nodes() > 0

    result = apply_structured_ib(graph, mode="gumbel_topk", config=IBConfig(seed=3, keep_ratio=0.5))
    subgraph = extract_subgraph(graph, result)
    assert subgraph.number_of_nodes() > 0
    assert subgraph.number_of_nodes() <= graph.number_of_nodes()
