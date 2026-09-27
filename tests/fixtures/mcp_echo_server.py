from mcp.server.fastmcp import FastMCP

server = FastMCP("AgentForge Echo")


@server.tool()
def echo(text: str) -> str:
    """Return the input text."""
    return f"echo:{text}"


if __name__ == "__main__":
    server.run()
