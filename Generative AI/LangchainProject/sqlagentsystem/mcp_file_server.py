from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP


def create_server(root_dir: Path) -> FastMCP:
    """Create an MCP server that exposes safe file read/write tools under root_dir."""
    root_dir = root_dir.resolve()
    mcp = FastMCP(name="local-files")

    def resolve_path(user_path: str) -> Path:
        candidate = (root_dir / user_path).resolve()
        if candidate != root_dir and root_dir not in candidate.parents:
            raise ValueError("Path escapes configured root directory")
        return candidate

    @mcp.tool()
    def list_dir(path: str = ".") -> str:
        target = resolve_path(path)
        if not target.exists():
            return f"Error: path not found: {target}"
        if not target.is_dir():
            return f"Error: not a directory: {target}"
        entries = sorted(p.name for p in target.iterdir())
        return "\\n".join(entries) if entries else "(empty)"

    @mcp.tool()
    def read_file(path: str) -> str:
        target = resolve_path(path)
        if not target.exists():
            return f"Error: file not found: {target}"
        if not target.is_file():
            return f"Error: not a file: {target}"
        return target.read_text(encoding="utf-8")

    @mcp.tool()
    def write_file(path: str, content: str, overwrite: bool = False) -> str:
        target = resolve_path(path)
        if target.exists() and not overwrite:
            return "Error: file already exists. Set overwrite=true to replace it."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target}"

    @mcp.tool()
    def append_file(path: str, content: str) -> str:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return f"Appended {len(content)} chars to {target}"

    return mcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local file MCP server")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory exposed to tools (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    server = create_server(root)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
