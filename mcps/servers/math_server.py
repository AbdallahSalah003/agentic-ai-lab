from mcp.server.fastmcp import FastMCP



mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two integer numbers"""
    return int(a) + int(b)

@mcp.tool()
def multiply(a: int, b: int) -> int: 
    """Multiply two integer numbers"""
    return int(a)*int(b)

if __name__=="__main__":
    mcp.run(transport="stdio")

    