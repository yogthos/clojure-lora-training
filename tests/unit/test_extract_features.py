"""Tests for synthetic data generation module: extract_features.py"""

import json
import pytest
from pathlib import Path
from src.codeflow.synthetic.extract_features import (
    ClojureFeature,
    _is_clojure_file,
    _read_file_safe,
    _parse_features,
    collect_clojure_files,
)


class TestIsClojureFile:
    def test_clj(self):
        assert _is_clojure_file("src/core.clj")

    def test_cljs(self):
        assert _is_clojure_file("src/ui/components.cljs")

    def test_cljc(self):
        assert _is_clojure_file("src/shared/util.cljc")

    def test_not_clojure(self):
        assert not _is_clojure_file("README.md")
        assert not _is_clojure_file("src/main.py")

    def test_test_files_excluded(self):
        assert not _is_clojure_file("test/core_test.clj")
        assert not _is_clojure_file("src/test/handler_test.clj")


class TestReadFileSafe:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.clj"
        f.write_text("(ns test)\n(def x 1)")
        content = _read_file_safe(f)
        assert "(ns test)" in (content or "")

    def test_nonexistent_file(self):
        assert _read_file_safe(Path("/nonexistent/clj/file.clj")) is None


class TestParseFeatures:
    def test_parses_dict_response(self):
        result = {"features": [
            {"feature_type": "macros", "name": "defmacro", "description": "test", "complexity": "simple"}
        ]}
        features = _parse_features(result, Path("/tmp"))
        assert len(features) == 1
        assert features[0].feature_type == "macros"

    def test_parses_json_string(self):
        result = json.dumps([
            {"feature_type": "atoms", "name": "atom", "description": "mutable state", "complexity": "simple"}
        ])
        features = _parse_features(result, Path("/tmp"))
        assert len(features) == 1
        assert features[0].name == "atom"

    def test_parses_markdown_code_block(self):
        result = '```json\n[{"feature_type": "spec", "name": "s/def", "description": "spec definition", "complexity": "moderate"}]\n```'
        features = _parse_features(result, Path("/tmp"))
        assert len(features) == 1
        assert features[0].feature_type == "spec"

    def test_handles_invalid_json(self):
        features = _parse_features("not json at all", Path("/tmp"))
        assert features == []


class TestClojureFeature:
    def test_from_dict(self):
        d = {
            "feature_type": "macros",
            "name": "my-macro",
            "description": "A custom macro",
            "file_path": "src/macros.clj",
            "line_hint": 42,
            "complexity": "complex",
        }
        feat = ClojureFeature.from_dict(d)
        assert feat.feature_type == "macros"
        assert feat.name == "my-macro"
        assert feat.description == "A custom macro"
        assert feat.file_path == "src/macros.clj"
        assert feat.line_hint == 42
        assert feat.complexity == "complex"

    def test_to_dict(self):
        feat = ClojureFeature(
            feature_type="protocols",
            name="Storage",
            description="Storage protocol",
            file_path="src/storage.clj",
            line_hint=10,
            complexity="moderate",
        )
        d = feat.to_dict()
        assert d["feature_type"] == "protocols"
        assert d["name"] == "Storage"
        assert "moderate" == d["complexity"]

    def test_from_dict_partial(self):
        feat = ClojureFeature.from_dict({"feature_type": "unknown"})
        assert feat.feature_type == "unknown"
        assert feat.name == ""
        assert feat.complexity == "simple"


class TestCollectClojureFiles:
    def test_collects_clj_files(self, tmp_path):
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "core.clj").write_text("(ns core)")
        (tmp_path / "src" / "utils.clj").write_text("(ns utils)")
        (tmp_path / "README.md").write_text("readme")

        files = collect_clojure_files(str(tmp_path))
        paths = [str(f.relative_to(tmp_path)) for f in files]
        assert "src/core.clj" in paths
        assert "src/utils.clj" in paths
        assert "README.md" not in paths

    def test_excludes_test_dirs(self, tmp_path):
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "test").mkdir(parents=True)
        (tmp_path / "src" / "core.clj").write_text("(ns core)")
        (tmp_path / "test" / "core_test.clj").write_text("(ns core-test)")

        files = collect_clojure_files(str(tmp_path))
        paths = [str(f.relative_to(tmp_path)) for f in files]
        assert "src/core.clj" in paths
        # test dirs should be excluded
        assert "test/core_test.clj" not in paths

    def test_empty_dir(self, tmp_path):
        files = collect_clojure_files(str(tmp_path))
        assert files == []
