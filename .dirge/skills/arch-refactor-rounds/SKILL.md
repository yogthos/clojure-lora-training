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
python -c "from changed.module import key_symbol; print('OK')"
```

After each round:
```bash
python -c "
import module1; print('OK')
import module2; print('OK')
...
"
```

Then run relevant tests:
```bash
python -m pytest tests/unit/test_changed_modules*.py -x -q --tb=short
```

## What NOT to do in early rounds

- Don't refactor logic — only move/rename/delete dead code
- Don't "clean up" code you're not targeting — stay focused on the task list
- Don't skip verification — one broken import cascades
- Don't do more than 5 tasks per round — verify and commit before continuing
