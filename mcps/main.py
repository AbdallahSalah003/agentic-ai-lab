import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters


load_dotenv()



llm = ChatOpenAI(
    base_url="https://backend.sovereigneg.com/v1",
    api_key=os.getenv("SOVEREIGNEG_API_KEY"),
    model="gpt-4.1-nano",
    temperature=0,
)

stdio_server_params = StdioServerParameters(
    command="python",
    args=['/home/abdallah/Learning/langchain_agenticai/mcps/servers/math_server.py']
)

async def main():
    async with stdio_client(stdio_server_params) as (read,write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize() # after initialization the mcp server exposed the tools to the client
            print("Client session is initialized")
            tools = await session.list_tools() # get tools from mcp client as (tuple of tools which is not compatible with langchain)
            list_of_tools = await load_mcp_tools(session) 
            agent = create_agent(llm,list_of_tools)
            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 21 + 2 * 77 equal?")]}) 
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
