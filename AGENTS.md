# Repository Guidelines

## Project Structure & Module Organization

This repository is an early Python prototype for a visual, step-level math-solution grading system. `main.py` is the current application entry point. `docs/` contains the system design and invention reference (`定稿--一种基于图文一致性的数学解答过程步骤级批改方法和系统.docx`). As the prototype grows, keep application modules, tests, and static assets in clearly named top-level directories such as `src/`, `tests/`, and `assets/`; avoid mixing uploaded student materials with source code.

## Build, Test, and Development Commands

There is no configured build system or test suite yet. Use the Python interpreter to run the prototype:

- `python main.py` — start the current entry point.
- `python -m compileall .` — check Python files for syntax errors.

When adding a UI or rendering stack, document its setup and run commands here and pin dependencies in a project manifest (for example, `requirements.txt` or `pyproject.toml`).

## Coding Style & Naming Conventions

Use Python 4-space indentation and standard-library conventions. Name modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Keep image analysis, step-order/logic analysis, error classification, geometry-region overlays, and feedback/report generation in separable modules so each can be developed and reviewed independently. Add type hints and concise docstrings to public interfaces.

## Testing Guidelines

No testing framework or coverage requirement is configured. Place future tests under `tests/`, mirror source module names, and use `test_*.py` naming (for example, `tests/test_step_analysis.py`). Include representative geometry, student-step, and expected-region cases when adding grading behavior. Run the project’s chosen test runner and syntax check before submitting changes.

## Commit & Pull Request Guidelines

Git history is not available in this checkout, so no repository-specific commit convention can be confirmed. Use short, imperative commit subjects (for example, `Add geometry error overlay`). Pull requests should explain the behavior changed, note setup or data requirements, link related work when applicable, and include screenshots for visual or rendering changes.

## Security & Configuration

Keep credentials and private student data out of source control. Use environment-based configuration for secrets, and use synthetic or anonymized examples in tests and screenshots.
