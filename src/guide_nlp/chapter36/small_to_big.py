"""Small-to-big: search small chunks, return their parent (chapter 36)."""

from guide_nlp.chapter36.chunking import sentence_spans


def index_children(parents: list[dict]) -> tuple[list[dict], list[int]]:
    """One child per sentence of each parent; parent_of maps a child to its parent."""
    children, parent_of = [], []
    for pid, parent in enumerate(parents):
        for start, end in sentence_spans(parent["text"]):
            children.append(dict(start=start, end=end))
            parent_of.append(pid)
    return children, parent_of


def search_parents(child_ids: list[int], parent_of: list[int], k: int = 5) -> list[int]:
    found: list[int] = []
    for child in child_ids:  # best first
        if parent_of[child] not in found:
            found.append(parent_of[child])
        if len(found) == k:
            break
    return found


if __name__ == "__main__":
    parents = [{"text": "Première phrase. Deuxième phrase."}, {"text": "Une autre. Et encore une."}]
    children, parent_of = index_children(parents)
    print(len(children), parent_of, search_parents([3, 1, 2, 0], parent_of, k=2))
