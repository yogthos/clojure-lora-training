"""Near-duplicate detection (MinHash LSH), and its use in assembly and the split."""

import random

from src.codeflow.assembly.assembler import drop_near_duplicates
from src.codeflow.assembly.near_dup import near_duplicate_clusters, record_text
from src.codeflow.assembly.splitter import assert_no_leakage, stratified_split


def _code(seed: int, n: int = 60) -> str:
    rng = random.Random(seed)
    words = ["defn", "let", "map", "reduce", "assoc", "update", "fn", "when", "cond",
             "str", "first", "rest", "conj", "into", "filter", "keep", "get-in", "merge"]
    return "\n".join(f"+({rng.choice(words)} x{rng.randint(0, 999)} {rng.choice(words)} "
                     f"y{rng.randint(0, 999)})" for _ in range(n))


class TestClusters:
    def test_identical_and_lightly_edited_texts_cluster(self):
        a = _code(1)
        b = a.replace("x1", "x1 ") + "\n+(println :done)"
        c = _code(2)
        ids = near_duplicate_clusters([a, b, c], threshold=0.8)
        assert ids[0] == ids[1] != ids[2]

    def test_half_overlap_is_not_a_duplicate(self):
        a, b = _code(1, 80), _code(2, 80)
        half = "\n".join(a.splitlines()[:40] + b.splitlines()[40:])
        ids = near_duplicate_clusters([a, half], threshold=0.8)
        assert ids[0] != ids[1]

    def test_short_texts_compare_exactly(self):
        ids = near_duplicate_clusters(["+(inc x)", "+(inc x)", "+(dec x)"], threshold=0.8)
        assert ids[0] == ids[1] != ids[2]

    def test_record_text_covers_every_assistant_turn(self):
        conv = [{"from": "human", "value": "H"}, {"from": "gpt", "value": "A1"},
                {"from": "human", "value": "H2"}, {"from": "gpt", "value": "A2"}]
        text = record_text({"instruction": "do x", "output": "A1", "conversations": conv})
        assert "A1" in text and "A2" in text and "H2" not in text


def _rec(repo, body, instruction):
    return {"instruction": instruction, "input": "", "repo": repo, "source": "git",
            "output": f"diff --git a/a.clj b/a.clj\n--- a/a.clj\n+++ b/a.clj\n@@ -1 +1,60 @@\n{body}\n"}


class TestAssembly:
    def test_keeps_the_first_of_each_near_duplicate_cluster(self):
        # datahike re-uses datascript's history: the same change, another repo.
        a = _rec("datascript", _code(1), "fix entity lookup for refs")
        b = _rec("datahike", _code(1) + "\n+(comment)", "Fix entity lookup for refs.")
        c = _rec("datahike", _code(3), "add history index")
        kept, dropped = drop_near_duplicates([a, b, c])
        assert kept == [a, c] and dropped == 1


class TestSplit:
    def test_similar_records_stay_on_one_side(self):
        recs = []
        for i in range(40):
            recs.append(_rec(f"r{i}", _code(i), f"change number {i} in the lib"))
        # Same code under different repos and wording: below the drop
        # threshold after edits, but similar enough to leak.
        for i in range(10):
            body = "\n".join(_code(i).splitlines()[:45] + _code(1000 + i).splitlines()[:15])
            recs.append(_rec(f"fork{i}", body, f"port of fix {i}"))
        train, val = stratified_split(recs, val_frac=0.2, seed=3)
        assert_no_leakage(train, val)
        side = {r["repo"]: ("train" if r in train else "val") for r in recs}
        for i in range(10):
            assert side[f"r{i}"] == side[f"fork{i}"]
