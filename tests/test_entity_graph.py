from __future__ import annotations

from harness.entity_graph import EntityGraph, extract_entities


def test_extracts_identifier_shaped_names_in_first_mention_order() -> None:
    # Arrange
    text = "The access code for site_000_01 is the access code for site_000_00."

    # Act
    entities = extract_entities(text)

    # Assert
    assert entities == ["site_000_01", "site_000_00"]


def test_repeated_mentions_are_extracted_once() -> None:
    assert extract_entities("site_1_a and site_1_a again") == ["site_1_a"]


def test_plain_prose_has_no_entities() -> None:
    assert extract_entities("The access code is 4271.") == []


def test_an_item_mentioning_an_anchor_is_zero_hops_away() -> None:
    # Arrange
    graph = EntityGraph()
    graph.add(0, ["a_1"])

    # Act / Assert
    assert graph.item_hops(["a_1"], hops=0) == {0: 0}


def test_an_item_linked_through_a_shared_entity_is_one_hop_away() -> None:
    # Arrange
    graph = EntityGraph()
    graph.add(0, ["a_1", "b_1"])
    graph.add(1, ["b_1"])
    graph.add(2, ["c_1"])

    # Act
    hops = graph.item_hops(["a_1"], hops=1)

    # Assert
    assert hops == {0: 0, 1: 1}


def test_items_beyond_the_hop_limit_are_not_reached() -> None:
    # Arrange
    graph = EntityGraph()
    graph.add(0, ["a_1", "b_1"])
    graph.add(1, ["b_1", "c_1"])
    graph.add(2, ["c_1"])

    # Act / Assert
    assert graph.item_hops(["a_1"], hops=1) == {0: 0, 1: 1}
    assert graph.item_hops(["a_1"], hops=2) == {0: 0, 1: 1, 2: 2}


def test_unknown_anchors_reach_nothing() -> None:
    # Arrange
    graph = EntityGraph()
    graph.add(0, ["a_1"])

    # Act / Assert
    assert graph.item_hops(["z_9"], hops=3) == {}
    assert graph.known(["z_9", "a_1"]) == ["a_1"]


def test_clear_forgets_every_node() -> None:
    # Arrange
    graph = EntityGraph()
    graph.add(0, ["a_1"])

    # Act
    graph.clear()

    # Assert
    assert graph.item_hops(["a_1"], hops=1) == {}
