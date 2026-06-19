"""Tests for git diff parser."""

import pytest
from src.codeflow.git_mining.diff_parser import parse_diff, DiffFile, DiffHunk


class TestParseDiff:
    """Unit tests for unified diff parsing."""

    SINGLE_FILE_DIFF = """\
diff --git a/src/core.clj b/src/core.clj
index abc123..def456 100644
--- a/src/core.clj
+++ b/src/core.clj
@@ -10,6 +10,8 @@
 (ns myapp.core
   (:require [clojure.string :as str]))

+(defn greet [name]
+  (str "Hello, " name "!"))
+
 (defn -main [& args]
-  (println "Starting...")
+  (println "Starting app v2...")
   (println "Ready."))
"""

    MULTI_FILE_DIFF = """\
diff --git a/src/core.clj b/src/core.clj
index abc..def 100644
--- a/src/core.clj
+++ b/src/core.clj
@@ -1,3 +1,4 @@
+(ns myapp.core)
 (def x 1)
-(def y 2)
+(def y 3)
diff --git a/src/utils.clj b/src/utils.clj
new file mode 100644
index 0000000..abcd123
--- /dev/null
+++ b/src/utils.clj
@@ -0,0 +1,3 @@
+(ns myapp.utils)
+(defn add [a b] (+ a b))
"""

    NEW_FILE_DIFF = """\
diff --git a/src/new_file.clj b/src/new_file.clj
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/new_file.clj
@@ -0,0 +1,5 @@
+(ns myapp.new-file
+  (:require [clojure.string :as str]))
+
+(defn process [data]
+  (str/upper-case data))
"""

    DELETED_FILE_DIFF = """\
diff --git a/src/old_file.clj b/src/old_file.clj
deleted file mode 100644
index abc1234..0000000
--- a/src/old_file.clj
+++ /dev/null
@@ -1,5 +0,0 @@
-(ns myapp.old-file
-  (:require [clojure.string :as str]))
-
-(defn process [data]
-  (str/upper-case data))
"""

    def test_parses_single_file_diff(self):
        files = parse_diff(self.SINGLE_FILE_DIFF)
        assert len(files) == 1
        f = files[0]
        assert isinstance(f, DiffFile)
        assert f.path == "src/core.clj"
        assert f.change_type == "modified"
        assert len(f.hunks) == 1
        hunk = f.hunks[0]
        assert isinstance(hunk, DiffHunk)
        assert hunk.old_start == 10
        assert hunk.old_count == 6
        assert hunk.new_start == 10
        assert hunk.new_count == 8
        # Verify we have both added and removed lines
        assert any(l.startswith("+") and not l.startswith("+++") for l in hunk.lines)
        assert any(l.startswith("-") and not l.startswith("---") for l in hunk.lines)

    def test_parses_multi_file_diff(self):
        files = parse_diff(self.MULTI_FILE_DIFF)
        assert len(files) == 2
        assert files[0].path == "src/core.clj"
        assert files[0].change_type == "modified"
        assert files[1].path == "src/utils.clj"
        assert files[1].change_type == "added"

    def test_parses_new_file_diff(self):
        files = parse_diff(self.NEW_FILE_DIFF)
        assert len(files) == 1
        assert files[0].change_type == "added"
        assert len(files[0].hunks) == 1

    def test_parses_deleted_file_diff(self):
        files = parse_diff(self.DELETED_FILE_DIFF)
        assert len(files) == 1
        assert files[0].change_type == "deleted"

    def test_handles_empty_input(self):
        files = parse_diff("")
        assert files == []

    def test_handles_no_diff_header(self):
        files = parse_diff("just some text\nnot a diff\n")
        assert files == []

    def test_extracts_added_and_removed_lines(self):
        files = parse_diff(self.SINGLE_FILE_DIFF)
        hunk = files[0].hunks[0]
        added = [l[1:] for l in hunk.lines if l.startswith("+") and not l.startswith("+++")]
        removed = [l[1:] for l in hunk.lines if l.startswith("-") and not l.startswith("---")]
        assert any("defn greet" in l for l in added)
        assert any("println \"Starting..." in l for l in removed)

    def test_metadata_populated(self):
        files = parse_diff(self.SINGLE_FILE_DIFF)
        assert len(files[0].hunks) > 0
        assert all(isinstance(h, DiffHunk) for h in files[0].hunks)
