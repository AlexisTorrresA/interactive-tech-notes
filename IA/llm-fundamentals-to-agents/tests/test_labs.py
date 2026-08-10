from app.labs import cosine_similarity, neuron_step, rag_search, softmax


def test_softmax_sums_to_one():
    p = softmax([2, 1, 0], 1.0)
    assert abs(sum(p) - 1) < 1e-9
    assert p[0] > p[1] > p[2]


def test_cosine_identity():
    assert abs(cosine_similarity([1, 2], [1, 2]) - 1) < 1e-9


def test_neuron_step_returns_update():
    r = neuron_step(x1=1, x2=.5, w1=.8, w2=-.4, bias=.1, target=1, learning_rate=.3)
    assert 0 < r["prediction"] < 1
    assert "w1" in r["updated"]


def test_rag_returns_ranked_results():
    r = rag_search("¿Qué es RAG?", top_k=2)
    assert r["results"][0]["score"] >= r["results"][1]["score"]
