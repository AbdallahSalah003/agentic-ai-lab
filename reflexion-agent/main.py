from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, END, START
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS=2
EXECUTE_TOOLS="execute_tools"
REVISOR="revisor"
RESPONDER="responder"
LAST=-1

def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal[EXECUTE_TOOLS, END]:
    """Determine wether to continue or end based on iteration count"""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS


builder = StateGraph(MessagesState)
builder.add_node(RESPONDER, draft_node)
builder.add_node(REVISOR, revise_node)
builder.set_entry_point(RESPONDER)
builder.add_node(EXECUTE_TOOLS, execute_tools)
builder.add_edge(RESPONDER, EXECUTE_TOOLS)
builder.add_conditional_edges(REVISOR, event_loop, {
    END: END,
    EXECUTE_TOOLS: EXECUTE_TOOLS
})
builder.add_edge(EXECUTE_TOOLS, REVISOR)


app = builder.compile()
# app.get_graph().draw_mermaid_png(output_file_path="graph.png")


res = app.invoke(
    {
        "messages": [
            HumanMessage(content="Write about AI-Powered SOC / Autonomous soc problem domain, list startups that do that and raised capital.")
        ]
    }
)
last_message  = res["messages"][LAST]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])

print()
print()
print(res)