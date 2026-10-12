"""Context a mined transition shows besides the changed code: where the code
has been heading (recent history) and who uses it (related code)."""

import subprocess
from pathlib import Path

from src.codeflow.git_mining.context import (
    HistoryEntry,
    changed_form_names,
    file_history,
    form_name,
    format_history,
    format_related,
    related_forms,
)
from src.codeflow.git_mining.miner import MinedExample, mine_repository


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


def _init(root: Path):
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")


def _commit(root: Path, files: dict, msg: str, date: str = "2020-01-01T00:00:00"):
    for path, content in files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(root, "add", "-A")
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", msg], check=True,
                   capture_output=True, env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
                                             "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})


def _head(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD").strip()


class TestFormName:
    def test_def_forms(self):
        assert form_name("(defn foo [x] x)") == "foo"
        assert form_name("(defn- ^:private bar\n  [x] x)") == "bar"
        assert form_name("(defmethod area :circle [c] 1)") == "area :circle"
        assert form_name("(defprotocol Shape (area [s]))") == "Shape"
        assert form_name("(ns app.core (:require [x]))") == "ns"

    def test_other_forms_have_no_name(self):
        assert form_name("(comment (foo))") is None
        assert form_name("(println 1)") is None


class TestChangedFormNames:
    def test_names_the_forms_a_hunk_touches(self):
        content = "(ns a)\n\n(defn f [] 1)\n\n(defn g [] 2)\n\n(defn h [] 3)\n"
        diff = ("diff --git a/a.clj b/a.clj\n--- a/a.clj\n+++ b/a.clj\n"
                "@@ -5 +5 @@\n-(defn g [] 2)\n+(defn g [] 22)\n")
        assert changed_form_names(diff, {"a.clj": content}, side="old") == {"a.clj": ["g"]}


class TestFileHistory:
    def test_recent_commits_touching_the_files_oldest_first(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"src/a.clj": "(ns a)\n\n(defn f [] 1)\n"}, "add f", "2020-01-01T00:00:00")
        _commit(tmp_path, {"src/b.clj": "(ns b)\n"}, "unrelated b", "2020-02-01T00:00:00")
        _commit(tmp_path, {"src/a.clj": "(ns a)\n\n(defn f [] 1)\n\n(defn g [] 2)\n"},
                "add g", "2020-03-01T00:00:00")
        _commit(tmp_path, {"src/a.clj": "(ns a)\n\n(defn f [] 10)\n\n(defn g [] 2)\n"},
                "make f ten", "2020-04-01T00:00:00")
        hist = file_history(str(tmp_path), _head(tmp_path), ["src/a.clj"], n=2)
        assert [h.subject for h in hist] == ["add g", "make f ten"]
        assert hist[0].date == "2020-03-01"
        assert hist[0].forms == {"src/a.clj": ["g"]}
        assert hist[1].forms == {"src/a.clj": ["f"]}

    def test_format(self):
        text = format_history([HistoryEntry("2020-03-01", "add g", {"src/a.clj": ["g", "h"]}),
                               HistoryEntry("2020-04-01", "docs", {})])
        assert text == ("## Recent history of these files\n"
                        "- 2020-03-01 add g (src/a.clj: g, h)\n"
                        "- 2020-04-01 docs")
        assert format_history([]) == ""


class TestRelatedForms:
    def test_finds_callers_in_other_namespaces(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {
            "src/app/core.clj": "(ns app.core)\n\n(defn fetch [url] url)\n\n(defn other [] 1)\n",
            "src/app/web.clj": ("(ns app.web\n  (:require [app.core :as core]))\n\n"
                                "(defn handler [req]\n  (core/fetch (:url req)))\n\n"
                                "(defn unrelated [] 2)\n"),
            "src/app/cli.clj": "(ns app.cli)\n\n(defn fetch [] :local)\n",
        }, "initial")
        diff = ("diff --git a/src/app/core.clj b/src/app/core.clj\n--- a/src/app/core.clj\n"
                "+++ b/src/app/core.clj\n@@ -3 +3 @@\n-(defn fetch [url] url)\n"
                "+(defn fetch [url] (slurp url))\n")
        before = {"src/app/core.clj": "(ns app.core)\n\n(defn fetch [url] url)\n\n(defn other [] 1)\n"}
        rel = related_forms(str(tmp_path), _head(tmp_path), before, diff)
        assert list(rel) == ["src/app/web.clj"]
        assert "(defn handler" in rel["src/app/web.clj"]
        assert "unrelated" not in rel["src/app/web.clj"]

    def test_format(self):
        assert format_related({}) == ""
        assert format_related({"src/b.clj": "(defn h [] (a/f))"}) == (
            "## Related code (uses the changed definitions)\n"
            "### src/b.clj\n```clojure\n(defn h [] (a/f))\n```")


_BIG = "(ns big)\n\n" + "".join(f"(defn f-{i} [] {i})\n\n" for i in range(800))


class TestWholeFiles:
    def _ex(self, before):
        path, content = next(iter(before.items()))
        diff = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
                "@@ -3 +3 @@\n-(defn f-0 [] 0)\n+(defn f-0 [] 100)\n")
        return MinedExample(repo_name="r", instruction="x", before=before, after=before,
                            diff=diff, changed_files=[path])

    def test_small_file_is_shown_whole(self):
        content = "(ns small)\n\n(defn f-0 [] 0)\n\n(defn keep-me [] 1)\n"
        assert "keep-me" in self._ex({"small.clj": content}).to_dict()["input"]

    def test_large_file_shows_changed_forms(self):
        inp = self._ex({"big.clj": _BIG}).to_dict()["input"]
        assert "f-0" in inp and "f-500" not in inp


class TestMinedInput:
    def test_arc_input_carries_history_and_related_code(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"src/app/core.clj": "(ns app.core)\n\n(defn fetch [url] url)\n",
                           "src/app/web.clj": ("(ns app.web (:require [app.core :as c]))\n\n"
                                               "(defn handler [r] (c/fetch r))\n")},
                "initial core and web namespaces", "2020-01-01T00:00:00")
        _commit(tmp_path, {"src/app/core.clj": "(ns app.core)\n\n(defn fetch [url] (str url))\n"},
                "coerce fetch argument to a string", "2020-02-01T00:00:00")
        _commit(tmp_path, {"src/app/core.clj": "(ns app.core)\n\n(defn fetch [url] (slurp (str url)))\n"},
                "read the url contents in fetch", "2020-03-01T00:00:00")
        ex = mine_repository(str(tmp_path), lifecycle_window=(0.0, 1.0), triplet_span=1)[-1]
        inp = ex.to_dict()["input"]
        assert "## Recent history of these files\n" in inp
        assert "- 2020-02-01 coerce fetch argument to a string (src/app/core.clj: fetch)" in inp
        assert "read the url contents" not in inp          # the arc's own commit isn't history
        assert "## Related code" in inp and "(defn handler" in inp
        assert inp.index("## Recent history") < inp.index("### src/app/core.clj")
