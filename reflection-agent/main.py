from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from chains import reflect_chain, generate_chain
from langchain_core.messages import HumanMessage

load_dotenv()


GENERATE = "generate"
REFLECT = "reflect"
LAST = -1


def should_countinue(state: MessagesState) -> str:
    if len(state["messages"]) > 4:
        return END
    return REFLECT


def generation_node(state: MessagesState) -> MessagesState:
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessagesState) -> MessagesState:
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


flow = StateGraph(MessagesState)

flow.add_node(GENERATE, generation_node)
flow.set_entry_point(GENERATE)
flow.add_node(REFLECT, reflection_node)

flow.add_conditional_edges(GENERATE, should_countinue, {
    END: END,
    REFLECT: REFLECT
})
flow.add_edge(REFLECT, GENERATE)

app = flow.compile()
# app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__=="__main__":
    inputs = HumanMessage(content="""
        Make this tweet better:
        @LangChainAI
        - newly Tool Calling feature is really underrated.
        After a long wait, it's here making the implementation of agents accross different models
        with function calling- super easy.

        Made a video covering their newest blog post
    """)

    res = app.invoke({"messages": inputs})
