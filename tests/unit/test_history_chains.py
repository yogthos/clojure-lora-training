"""History chains: one namespace followed through consecutive changes, as a
multi-turn record (request -> diff -> next request on the code as it now
stands -> diff ...)."""

import json
import subprocess
from pathlib import Path

from src.codeflow.assembly.validator import validate_example
from src.codeflow.git_mining.miner import MinedChain, MinedExample, mine_repository


def _git(root: Path, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _init(root: Path):
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")


def _commit(root: Path, files: dict, msg: str):
    for path, content in files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def _a(*bodies):
    return "(ns a)\n\n" + "\n\n".join(f"(defn f{i} [] {b})" for i, b in enumerate(bodies)) + "\n"


def _evolving_repo(root: Path):
    _init(root)
    _commit(root, {"src/a.clj": _a(1, 2, 3, 4)}, "initial namespace with four functions")
    _commit(root, {"src/a.clj": _a(10, 2, 3, 4)}, "make f0 return ten instead of one")
    _commit(root, {"src/a.clj": _a(10, 20, 3, 4)}, "make f1 return twenty instead of two")
    _commit(root, {"src/a.clj": _a(10, 20, 30, 4)}, "make f2 return thirty instead of three")
    _commit(root, {"src/a.clj": _a(10, 20, 30, 40)}, "make f3 return forty instead of four")


def _mine(root, chain_max):
    return mine_repository(str(root), lifecycle_window=(0.0, 1.0), triplet_span=1,
                           chain_max=chain_max)


class TestChainBuilding:
    def test_consecutive_changes_to_one_file_chain(self, tmp_path):
        _evolving_repo(tmp_path)
        out = _mine(tmp_path, chain_max=3)
        assert [type(r).__name__ for r in out] == ["MinedChain", "MinedChain"]
        assert [len(c.turns) for c in out] == [3, 2]

    def test_chain_max_one_keeps_independent_arcs(self, tmp_path):
        _evolving_repo(tmp_path)
        out = _mine(tmp_path, chain_max=1)
        assert all(isinstance(r, MinedExample) for r in out) and len(out) == 5

    def test_unrelated_files_break_the_chain(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"src/a.clj": _a(1, 2), "src/b.clj": "(ns b)\n\n(defn g [] 1)\n"},
                "initial namespaces a and b")
        _commit(tmp_path, {"src/a.clj": _a(10, 2)}, "make f0 return ten instead of one")
        _commit(tmp_path, {"src/b.clj": "(ns b)\n\n(defn g [] 100)\n"}, "make g return one hundred")
        _commit(tmp_path, {"src/a.clj": _a(10, 20)}, "make f1 return twenty instead of two")
        out = _mine(tmp_path, chain_max=4)
        # a's changes are not consecutive arcs, so nothing links across b's change.
        assert not any(isinstance(r, MinedChain) and len(r.turns) > 2 for r in out)
        chained = [r for r in out if isinstance(r, MinedChain)]
        for c in chained:
            files = [set(t.changed_files) for t in c.turns]
            assert all(files[i] & files[i + 1] for i in range(len(files) - 1))

    def test_unshown_change_in_between_breaks_the_chain(self, tmp_path):
        # A rejected commit ("wip") changes a.clj between two arcs: the second
        # arc's code isn't what the first left behind.
        _init(tmp_path)
        _commit(tmp_path, {"src/a.clj": _a(1, 2, 3)}, "initial namespace with three functions")
        _commit(tmp_path, {"src/a.clj": _a(10, 2, 3)}, "make f0 return ten instead of one")
        _commit(tmp_path, {"src/a.clj": _a(10, 2, 33)}, "wip")
        _commit(tmp_path, {"src/a.clj": _a(10, 20, 33)}, "make f1 return twenty instead of two")
        out = _mine(tmp_path, chain_max=4)
        for c in (r for r in out if isinstance(r, MinedChain)):
            subjects = [t.instruction for t in c.turns]
            assert not ("make f0 return ten instead of one" in subjects
                        and "make f1 return twenty instead of two" in subjects)

    def test_size_budget_caps_a_chain(self, tmp_path):
        _evolving_repo(tmp_path)
        whole = _mine(tmp_path, chain_max=5)
        capped = mine_repository(str(tmp_path), lifecycle_window=(0.0, 1.0), triplet_span=1,
                                 chain_max=5, chain_chars=900)
        assert len(capped) > len(whole)
        for c in (r for r in capped if isinstance(r, MinedChain)):
            assert sum(len(t["value"]) for t in c.to_dict()["conversations"]) <= 900


class TestChainRecord:
    def test_turns_alternate_and_later_turns_show_the_current_code(self, tmp_path):
        _evolving_repo(tmp_path)
        chain = _mine(tmp_path, chain_max=3)[0]
        rec = chain.to_dict()
        roles = [t["from"] for t in rec["conversations"]]
        assert roles == ["human", "gpt"] * 3
        third = rec["conversations"][4]["value"]
        assert third.startswith("make f1 return twenty instead of two\n")
        assert "(defn f0 [] 10)" in third            # the previous change is in place
        assert "## Recent history" not in third      # the conversation is the history
        assert rec["conversations"][1]["value"].startswith("diff --git ")
        # Flat fields mirror turn one, for dedup and the grouped split.
        assert rec["instruction"] == "initial namespace with four functions"
        assert rec["output"] == rec["conversations"][1]["value"]
        assert rec["repo"] == chain.repo_name


def _chain_record(think=(True, True)):
    diff = ("diff --git a/src/a.clj b/src/a.clj\n--- a/src/a.clj\n+++ b/src/a.clj\n"
            "@@ -1,2 +1,2 @@\n (ns a)\n-(defn f [] 1)\n+(defn f [] 2)\n")
    gpt = [("<think>\nf should return two now.\n</think>\n\n" if t else "") + diff for t in think]
    conv = [{"from": "human", "value": "make f return two\n### src/a.clj"}, {"from": "gpt", "value": gpt[0]},
            {"from": "human", "value": "make f return two again\n### src/a.clj"}, {"from": "gpt", "value": gpt[1]}]
    return {"instruction": "make f return two", "input": "### src/a.clj", "output": gpt[0],
            "conversations": conv}


class TestChainValidation:
    def test_valid_chain(self):
        r = validate_example(_chain_record())
        assert r.is_valid, r.to_dict()

    def test_every_turn_needs_reasoning(self):
        r = validate_example(_chain_record(think=(True, False)))
        assert not r.is_valid
        assert any("turn 3" in e for e in r.format_errors)

    def test_every_turn_is_checked_as_a_diff(self):
        rec = _chain_record()
        rec["conversations"][3]["value"] = "<think>\nok.\n</think>\n\nnot a diff"
        assert not validate_example(rec).is_valid
