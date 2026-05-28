from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.runtime import get_runtime

from sqlagentsystem.config import load_settings
from sqlagentsystem.memory import MemoryManager

SYSTEM_PROMPT = """You are a careful SQLite analyst with file tool access.

Rules:
- Think step-by-step.
- When database data is needed, call `execute_sql` directly. Do not ask for permission.
- SQL must be read-only.
- Use MCP file tools when the user asks to read/write files.
- Keep answers concise and include evidence from tool results.
"""

@dataclass
class RuntimeContext:
    db: SQLDatabase


@tool
def execute_sql(query: str) -> str:
    """Execute one read-only SQLite SELECT query and return results."""
    sql = query.strip()
    if not sql.lower().startswith("select"):
        return "Error: Only SELECT queries are allowed."
    if ";" in sql[:-1]:
        return "Error: Only one query at a time is allowed."

    runtime = get_runtime(RuntimeContext)
    try:
        return runtime.context.db.run(sql)
    except Exception as exc:
        return f"Error: {exc}"


async def load_mcp_tools(files_root: Path) -> list:
    server_script = (Path(__file__).parent / "mcp_file_server.py").resolve()
    client = MultiServerMCPClient(
        connections={
            "local_files": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(server_script), "--root", str(files_root)],
                "cwd": str(files_root),
            }
        },
        tool_name_prefix=True,
    )
    return await client.get_tools()


def latest_ai_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            return str(content)
    return "(no AI response found)"


async def run_chat() -> None:
    settings = load_settings()
    if not settings.db_path.exists():
        raise FileNotFoundError(f"DB file not found: {settings.db_path}")

    db = SQLDatabase.from_uri(f"sqlite:///{settings.db_path}")
    mcp_tools = await load_mcp_tools(files_root=Path.cwd())
    memory = MemoryManager(
        max_messages=50,
        memory_file=Path.cwd() / "sqlagentsystem" / "data" / "memory.jsonl",
    )

    agent = create_agent(
        model=settings.model_name,
        tools=[execute_sql, *mcp_tools],
        system_prompt=SYSTEM_PROMPT,
        context_schema=RuntimeContext,
    )

    print("SQL Agent System")
    print("Type 'exit' to quit.")

    while True:
        user_text = input("User> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("bye")
            break

        recalled = memory.recall(user_text, k=4)
        augmented_input = user_text
        if recalled:
            memory_block = "\n".join(f"- {item}" for item in recalled)
            augmented_input = (
                f"{user_text}\n\n"
                "Relevant past context (use only if helpful):\n"
                f"{memory_block}"
            )

        current_messages = memory.short_term_messages() + [HumanMessage(content=augmented_input)]
        result = await agent.ainvoke(
            {"messages": current_messages},
            context=RuntimeContext(db=db),
        )
        response_messages = result.get("messages", current_messages)
        ai_text = latest_ai_text(response_messages)
        print(f"agent> {ai_text}")
        memory.add_turn(user_text=user_text, agent_text=ai_text)


def main() -> None:
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
