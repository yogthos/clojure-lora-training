"""Tests for unified-diff integrity checks and hunk-header repair."""

from src.codeflow.diffs import (
    apply_section_errors,
    check_patch,
    extract_diff,
    fix_hunk_headers,
    hunk_count_errors,
)

GOOD = (
    "diff --git a/x.clj b/x.clj\n"
    "--- a/x.clj\n"
    "+++ b/x.clj\n"
    "@@ -1,2 +1,2 @@\n"
    " (ns x)\n"
    "-(defn f [] 1)\n"
    "+(defn f [] 2)\n"
)


class TestHunkCounts:
    def test_valid_diff_has_no_errors(self):
        assert hunk_count_errors(GOOD) == []
        assert check_patch(GOOD) is None

    def test_header_claims_more_lines_than_body(self):
        bad = GOOD.replace("@@ -1,2 +1,2 @@", "@@ -1,3 +1,3 @@")
        assert hunk_count_errors(bad)
        assert check_patch(bad) is not None

    def test_header_claims_fewer_lines_than_body(self):
        bad = GOOD.replace("@@ -1,2 +1,2 @@", "@@ -1,1 +1,1 @@")
        assert hunk_count_errors(bad)
        assert check_patch(bad) is not None

    def test_truncated_header_is_rejected(self):
        assert check_patch(GOOD + "@@ -1,0 +") is not None

    def test_omitted_count_means_one(self):
        d = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
        assert hunk_count_errors(d) == []

    def test_multiple_files_and_hunks(self):
        d = GOOD + GOOD.replace("x.clj", "y.clj")
        assert hunk_count_errors(d) == []
        assert check_patch(d) is None

    def test_empty_diff_is_rejected(self):
        assert check_patch("") is not None


class TestFixHunkHeaders:
    def test_recomputes_wrong_counts(self):
        bad = GOOD.replace("@@ -1,2 +1,2 @@", "@@ -1,7 +1,30 @@")
        fixed = fix_hunk_headers(bad)
        assert "@@ -1,2 +1,2 @@" in fixed
        assert check_patch(fixed) is None

    def test_bare_blank_lines_inside_hunk_become_context(self):
        d = (
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n"
            "@@ -1,9 +1,9 @@\n (a)\n\n-(b)\n+(c)\n"
        )
        fixed = fix_hunk_headers(d)
        assert "@@ -1,3 +1,3 @@\n (a)\n \n-(b)\n+(c)" in fixed
        assert check_patch(fixed) is None

    def test_trailing_blank_lines_are_not_counted(self):
        d = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n\n\n"
        assert "@@ -1,1 +1,1 @@" in fix_hunk_headers(d)

    def test_keeps_section_heading_after_header(self):
        d = GOOD.replace("@@ -1,2 +1,2 @@", "@@ -1,5 +1,5 @@ (defn f")
        assert "@@ -1,2 +1,2 @@ (defn f" in fix_hunk_headers(d)

    def test_valid_diff_unchanged(self):
        assert fix_hunk_headers(GOOD) == GOOD


class TestApplySection:
    TRACE = ";; eval: (+ 1 2)\n;; result: 3\n"

    def test_trace_with_apply_and_diff_is_ok(self):
        assert apply_section_errors(self.TRACE + ";; apply:\n" + GOOD) == []

    def test_missing_apply_marker(self):
        assert apply_section_errors(self.TRACE + GOOD)

    def test_bare_apply_marker_is_rejected(self):
        assert apply_section_errors(self.TRACE + "apply:\n" + GOOD)

    def test_apply_without_diff(self):
        assert apply_section_errors(self.TRACE + ";; apply:\n")

    def test_plain_diff_output_needs_no_marker(self):
        assert apply_section_errors(GOOD) == []

    def test_extract_diff_starts_at_header(self):
        assert extract_diff(self.TRACE + ";; apply:\n" + GOOD) == GOOD
        assert extract_diff(self.TRACE) == ""


class TestLineSplitting:
    """git counts lines by \\n only; form feeds (Lisp page breaks) and \\r are content."""

    FF = (
        "diff --git a/x.clj b/x.clj\n--- a/x.clj\n+++ b/x.clj\n"
        "@@ -1,3 +1,3 @@\n (ns x)\n \x0c\n-(def a 1)\n+(def a 2)\n"
    )

    def test_form_feed_line_counts_once(self):
        assert hunk_count_errors(self.FF) == []
        assert check_patch(self.FF) is None

    def test_fix_preserves_form_feed_and_trailing_newline(self):
        assert fix_hunk_headers(self.FF) == self.FF

    def test_crlf_content_counts_once(self):
        d = self.FF.replace(" \x0c\n", " (comment)\r\n")
        assert hunk_count_errors(d) == []
