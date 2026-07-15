# model

Hugging Face 模型应用（"hugging face模型应用"）。

## Cursor Cloud specific instructions

This section captures durable, non-obvious context for future cloud agents.

### Current repository state

- This repo is an **early Python scaffold**. As of setup it contains only `README.md`, `.gitignore`, and this file — **no source code, no dependency manifest, and no runnable services yet**.
- Intended stack (inferred from `README.md` + the Python `.gitignore`): a **Python 3.12 Hugging Face model application**.

### Environment

- Python is **3.12** (`/usr/bin/python3`). There is no `python` alias — use `python3`.
- The system package `python3.12-venv` is required for `venv`/`ensurepip` and is installed in the VM snapshot (not in the update script, since it is a system dependency).
- A project virtualenv lives at `.venv` (git-ignored). Use it for all dev work: run tools via `.venv/bin/python` / `.venv/bin/pip`, or `source .venv/bin/activate`.
- The startup **update script** runs `python3 -m venv .venv` (idempotent) and, **only if present**, installs from `requirements.txt` and/or `pyproject.toml` into `.venv`. When no manifest exists (current state) it just ensures the venv.

### Adding code / dependencies

- Declare dependencies in `requirements.txt` (or `pyproject.toml`); the update script will auto-install them into `.venv` on the next startup. To install immediately in-session: `.venv/bin/pip install -r requirements.txt`.
- Hugging Face libraries are large (`torch`, `transformers`); `huggingface_hub` (the lightweight Hub client) was verified working and can reach the public Hub.

### Lint / test / build / run

- **None configured yet** — there are no lint, test, build, or run commands until source code and tooling (e.g. `ruff`, `pytest`, an entrypoint/app) are added. Add them here (or reference `pyproject.toml`/`Makefile`) once they exist.
