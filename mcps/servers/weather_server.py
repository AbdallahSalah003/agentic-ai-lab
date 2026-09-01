from mcp.server.fastmcp import FastMCP
from typing import List

mcp = FastMCP("Weather")



@mcp.tool()
async def get_weather(location: str)  -> str:
    """Get a weather for a location."""
    return f"It's sunny in {location}."


if __name__=="__main__":
    mcp.run(transport="sse")


    