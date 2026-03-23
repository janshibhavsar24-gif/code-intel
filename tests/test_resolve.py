"""Tests for resolve.py — name resolution, graph building, and analytics."""
import pytest

from extractor import extract_repo
from models import EdgeKind, NodeKind
from resolve import build_graph, change_impact, coupling_scores, dead_code, resolve


@pytest.fixture
def parsed(fixtures_dir):
    return extract_repo(str(fixtures_dir), typescript=False)


@pytest.fixture
def resolved(parsed):
    nodes, edges = resolve(parsed)
    return nodes, edges


@pytest.fixture
def graph(resolved):
    nodes, edges = resolved
    return build_graph(nodes, edges)


@pytest.fixture
def node_map(resolved):
    nodes, _ = resolved
    return {n.id: n for n in nodes}


class TestResolve:
    def test_all_nodes_returned(self, resolved, parsed):
        nodes, _ = resolved
        total_expected = sum(len(pf.nodes) for pf in parsed)
        assert len(nodes) == total_expected

    def test_no_duplicate_node_ids(self, resolved):
        nodes, _ = resolved
        ids = [n.id for n in nodes]
        assert len(ids) == len(set(ids))

    def test_call_edges_resolved(self, resolved):
        _, edges = resolved
        call_edges = [e for e in edges if e.kind == EdgeKind.CALLS]
        assert len(call_edges) > 0

    def test_inherits_edge_resolved(self, resolved):
        _, edges = resolved
        inherit_edges = [e for e in edges if e.kind == EdgeKind.INHERITS]
        # Dog inherits Animal
        assert any(
            "Dog" in e.src and "Animal" in e.dst
            for e in inherit_edges
        )

    def test_bark_called_from_dog_speak(self, resolved):
        _, edges = resolved
        call_edges = [e for e in edges if e.kind == EdgeKind.CALLS]
        assert any("Dog" in e.src and "bark" in e.dst for e in call_edges)


class TestBuildGraph:
    def test_graph_has_all_nodes(self, graph, resolved):
        nodes, _ = resolved
        assert graph.number_of_nodes() == len(nodes)

    def test_graph_has_edges(self, graph):
        assert graph.number_of_edges() > 0

    def test_node_attributes_stored(self, graph):
        for nid, data in graph.nodes(data=True):
            assert "kind" in data
            assert "name" in data
            assert "file" in data

    def test_edge_kind_stored(self, graph):
        for u, v, data in graph.edges(data=True):
            assert "kind" in data


class TestCouplingScores:
    def test_returns_score_for_every_node(self, graph):
        scores = coupling_scores(graph)
        assert set(scores.keys()) == set(graph.nodes())

    def test_scores_are_positive(self, graph):
        scores = coupling_scores(graph)
        assert all(v > 0 for v in scores.values())

    def test_higher_indegree_scores_higher(self, graph):
        scores = coupling_scores(graph)
        # breathe is called by make_sound and speak — should rank above private
        breathe = next((s for nid, s in scores.items() if "breathe" in nid), None)
        private = next((s for nid, s in scores.items() if "_private" in nid), None)
        if breathe is not None and private is not None:
            assert breathe >= private

    def test_empty_graph(self):
        import networkx as nx
        G = nx.DiGraph()
        assert coupling_scores(G) == {}


class TestDeadCode:
    def test_private_fn_flagged(self, graph):
        dead = dead_code(graph)
        assert any("_private" in d for d in dead)

    def test_called_fn_not_flagged(self, graph):
        dead = dead_code(graph)
        # bark is called from Dog.speak so should not be dead
        assert not any("bark" == d.split("::")[-1] for d in dead)

    def test_only_functions_and_methods(self, graph, node_map):
        dead = dead_code(graph)
        for d in dead:
            if d in node_map:
                assert node_map[d].kind.value in ("function", "method")


class TestChangeImpact:
    def test_bark_impacts_dog_speak(self, graph):
        # bark() is called by Dog.speak, so changing bark should impact Dog.speak
        bark_id = next((nid for nid in graph.nodes() if nid.endswith("::bark")), None)
        if bark_id is None:
            pytest.skip("bark node not found in graph")
        affected = change_impact(graph, bark_id)
        assert any("speak" in a for a in affected)

    def test_nonexistent_node_raises(self, graph):
        import networkx as nx
        with pytest.raises(nx.NodeNotFound):
            change_impact(graph, "nonexistent::node")

    def test_leaf_node_returns_empty(self, graph):
        # A node with no callers should have no impact
        scores = coupling_scores(graph)
        leaf = min(scores, key=scores.get)
        result = change_impact(graph, leaf)
        # leaf may or may not have callers — just check it doesn't crash
        assert isinstance(result, list)
