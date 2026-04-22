# sqlagentsystem

Standalone SQL agent runtime (separate from notebooks) with:

- Interactive terminal chat loop
- Conversation memory across turns in-session
- Short-term memory window (last 50 messages)
- Long-term memory persisted to JSONL + semantic recall via embeddings
- SQLite querying tool (`execute_sql`)
- MCP-backed local file tools (`list_dir`, `read_file`, `write_file`, `append_file`)

## Setup

Use your existing project virtualenv and environment variables:

- `OPENAI_API_KEY` (required)
- `LANGSMITH_API_KEY` (optional)
- `LANGSMITH_ENDPOINT` (optional, set if tracing enabled)
- `LANGSMITH_TRACING_V2` (optional)
- `MODEL_NAME` (optional, default `openai:gpt-5-mini`)
- `DB_PATH` (optional, default `Chinook.db`)

## Run

From project root:

```bash
.venv/bin/python -m sqlagentsystem.chat_sql_agent
```

## Notes

- The MCP file server is started automatically as a stdio subprocess.
- File access is sandboxed to the current working directory used when you launch the script.
- Long-term memory file: `sqlagentsystem/data/memory.jsonl`
- The notebook files are not modified by this module.
