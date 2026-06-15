"""Tests for diff pattern classifier."""

import pytest
from src.git_mining.pattern_classifier import (
    classify_diff,
    DiffClassification,
    _is_clojure_source,
)


class TestIsClojureSource:
    def test_clj(self):
        assert _is_clojure_source("src/core.clj")

    def test_cljs(self):
        assert _is_clojure_source("src/ui/components.cljs")

    def test_cljc(self):
        assert _is_clojure_source("src/shared/util.cljc")

    def test_edn(self):
        assert _is_clojure_source("config/settings.edn")

    def test_not_clojure(self):
        assert not _is_clojure_source("README.md")
        assert not _is_clojure_source("src/core.py")


class TestPureRefactor:
    def test_simple_refactor(self):
        diff = """-(map #(* % 2) coll)
+(map (partial * 2) coll)"""
        result = classify_diff(diff, ["src/core.clj"])
        assert result.is_pure_refactor
        assert "pure-refactor" in result.patterns_found

    def test_not_pure_if_side_effect(self):
        diff = """
+  (spit "output.edn" (pr-str data))
+  (println "done")"""
        result = classify_diff(diff, ["src/core.clj"])
        assert not result.is_pure_refactor

    def test_not_pure_if_non_source_files(self):
        diff = """-(map f coll)
+(comp f g coll)"""
        result = classify_diff(diff, ["README.md", "config.edn"])
        assert not result.is_pure_refactor


class TestStateMachine:
    def test_atom_with_swap(self):
        diff = """
+(def state (atom {:count 0}))
+(swap! state update :count inc)"""
        result = classify_diff(diff, ["src/state.clj"])
        assert result.is_state_machine

    def test_no_atom(self):
        diff = "-(def x 1)\n+(def x 2)"
        result = classify_diff(diff, ["src/core.clj"])
        assert not result.is_state_machine

    def test_dosync_ref(self):
        diff = """
+(dosync
+  (alter account-balance + amount))"""
        result = classify_diff(diff, ["src/account.clj"])
        assert result.is_state_machine


class TestSideEffectIsolation:
    def test_effects_file_with_io(self):
        diff = """
+(defn save-user [user]
+  (spit "users.edn" (pr-str user)))"""
        result = classify_diff(diff, ["src/effects/io.clj"])
        assert result.is_side_effect_isolation

    def test_core_file_not_isolated(self):
        diff = """
+(defn save-user [user]
+  (spit "users.edn" (pr-str user)))"""
        result = classify_diff(diff, ["src/core.clj"])
        assert not result.is_side_effect_isolation


class TestMacro:
    def test_defmacro_added(self):
        diff = "+(defmacro when-let* [bindings & body] ...)"
        result = classify_diff(diff, ["src/macros.clj"])
        assert result.is_macro_change

    def test_no_macro(self):
        diff = "+(defn add [a b] (+ a b))"
        result = classify_diff(diff, ["src/core.clj"])
        assert not result.is_macro_change


class TestProtocol:
    def test_defprotocol(self):
        diff = "+(defprotocol Storage (save! [this data]) (load! [this id]))"
        result = classify_diff(diff, ["src/storage.clj"])
        assert result.is_protocol_change

    def test_defrecord(self):
        diff = "+(defrecord MemoryStorage [] Storage (save! [this data] ...))"
        result = classify_diff(diff, ["src/storage.clj"])
        assert result.is_protocol_change

    def test_extend_type(self):
        diff = "+(extend-type String Storage (save! [this data] ...))"
        result = classify_diff(diff, ["src/storage.clj"])
        assert result.is_protocol_change


class TestSpec:
    def test_sdef(self):
        diff = "+(s/def ::email string?)"
        result = classify_diff(diff, ["src/specs.clj"])
        assert result.is_spec_change

    def test_fdef(self):
        diff = "+(s/fdef validate :args (s/cat :data map?) :ret boolean?)"
        result = classify_diff(diff, ["src/specs.clj"])
        assert result.is_spec_change


class TestAsync:
    def test_go_loop(self):
        diff = """
+(go-loop [events (chan)]
+  (when-let [e (<! events)]
+    (process e)
+    (recur events)))"""
        result = classify_diff(diff, ["src/worker.clj"])
        assert result.is_async_change

    def test_no_async(self):
        diff = "+(defn process [e] (update e :status inc))"
        result = classify_diff(diff, ["src/worker.clj"])
        assert not result.is_async_change


class TestMultimethod:
    def test_defmulti(self):
        diff = "+(defmulti handle-event :type)"
        result = classify_diff(diff, ["src/events.clj"])
        assert result.is_multimethod_change

    def test_defmethod(self):
        diff = "+(defmethod handle-event :create [event] ...)"
        result = classify_diff(diff, ["src/events.clj"])
        assert result.is_multimethod_change


class TestMultiplePatterns:
    def test_spec_and_protocol(self):
        diff = """
+(defprotocol Validatable (validate [this]))
+(s/def ::name string?)"""
        result = classify_diff(diff, ["src/validation.clj"])
        assert result.is_protocol_change
        assert result.is_spec_change
        assert len(result.patterns_found) >= 2
