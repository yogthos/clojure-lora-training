---
name: arch-refactor-rounds
description: Execute architectural refactoring in small, verified rounds. Each round targets 3-5 concrete, testable changes with import and test verification.
---

# Architectural Refactoring in Rounds

When cleaning up a codebase, work in **small rounds** of 3-5 concrete tasks. Each round is independently verifiable — imports work, tests pass. This keeps the codebase functional at all times and prevents large, unverifiable diffs.

## Round Structure

Each round:
1. **3-5 concrete tasks** — specific files, specific changes, no ambiguity
2. **One task at a time** — make the change, verify imports, move to next
3. **End-of-round verification** — import all changed modules + run relevant tests

## Task Categories (in priority order)

Priority order for early rounds:
1. **Create missing shared modules** (models, types, constants that multiple modules import but that don't exist)
2. **Remove noise** ([LEGACY] markers, empty files, dead docstrings)
3. **Centralize scattered concerns** (env vars, config loading, logging setup)
4. **Fix import chains** (ensure __init__.py files export what's actually used)

Later rounds:
5. **Consolidate duplication** (merge similar classes, extract shared base)
6. **Fix design patterns** (god objects, feature envy, tight coupling)
7. **Reorganize module boundaries** (move files between packages)

## Verification Pattern

After each task:
```bash
uv run python -c "from changed.module import key_symbol; print('OK')"
```

After each round:
```bash
uv run python -c "
import module1; print('OK')
import module2; print('OK')
...
"
```

Then run relevant tests. Add `-p no:cacheprovider` to every pytest invocation during a refactor — it disables the on-disk cache so a stale `.pytest_cache` can never mask an import failure or reuse a stale collected test:
```bash
uv run python -m pytest tests/unit/test_changed_modules*.py -x -q --tb=short -p no:cacheprovider
```

**Full-suite-per-round (when the suite is fast):** on this project the full `tests/unit` suite (~837 tests, 5 skipped) runs quickly, so prefer running the FULL suite every round rather than just the changed modules — stronger regression coverage, and it's the same command as the session-end gate:
```bash
uv run python -m pytest tests/unit -q --tb=short -p no:cacheprovider
```

**Session-end gate (non-negotiable):** before declaring the refactor done, run the FULL unit suite and require zero failures. If spaCy (`en_core_web_sm`) is NOT installed, add `--ignore=tests/unit/test_content_classifier.py --ignore=tests/unit/test_transfer.py` to drop the spaCy-dependent tests (see Pitfalls); if spaCy IS installed, run everything.

## Pitfalls

- `git push origin main` may require human approval via the approval_provider. If blocked, tell the user explicitly and do not consider the session complete.
- Tests in `test_content_classifier.py` and `test_transfer.py` require spaCy (`en_core_web_sm`). When spaCy isn't installed, skip those with `--ignore=tests/unit/test_content_classifier.py --ignore=tests/unit/test_transfer.py`.

## What NOT to do in early rounds

- Don't refactor logic — only move/rename/delete dead code
- Don't "clean up" code you're not targeting — stay focused on the task list
- Don't skip verification — one broken import cascades
- Don't do more than 5 tasks per round — verify and commit before continuing
