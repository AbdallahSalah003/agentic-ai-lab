from typing import List
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()


tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: List[str], **kwargs):
    """Run the generated search queries"""
    return tavily_tool.batch([{"query": query} for query in search_queries])

# ToolNode will check state["messages"] and look at the last AIMessage
# if there are tool calls it will execute them and return the new updated state
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
