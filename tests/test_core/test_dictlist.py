import pytest

from synonim import DictList, Feature


def test_dictlist_remove_updates_id_index_for_object_and_id():
    first = Feature("f1")
    second = Feature("f2")
    third = Feature("f3")
    items = DictList([first, second, third])

    items.remove(second)

    assert "f2" not in items
    assert items.index("f3") == 1
    assert items["f3"] is third

    items.remove("f1")

    assert [item.id for item in items] == ["f3"]
    assert items.index(third) == 0


def test_dictlist_index_rejects_different_object_with_same_id():
    original = Feature("f1")
    duplicate = Feature("f1")
    items = DictList([original])

    with pytest.raises(ValueError, match="identical id"):
        items.index(duplicate)
