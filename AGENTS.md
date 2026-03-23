Documentation TOCs (start with the one relevant to your task):
- `README.md` shows basic usage
- `docs/HighJax docs.md` — Top-level TOC
- `docs/HighJax/HighJax docs.md` — HighJax environment (state, observations, reward, NPCs)
- `docs/Octane/Octane docs.md` — Octane TUI explorer (navigation, key bindings, rendering)

More things:
- Critical for writing code: `docs/HighJax coding conventions.md` Don't write code without consulting this and abiding to it.
- Run tests like `JAX_PLATFORMS=cpu pytest -n 12 <...other pytest args if needed>`
