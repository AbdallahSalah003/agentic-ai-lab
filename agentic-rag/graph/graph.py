import os
from dotenv import load_dotenv
from graph.nodes import generate, web_search, retrieve, grade_documents
from graph.chains.hallucinations_grader import hallucinations_grader_chain
from graph.chains.answer_grader import answer_grader_chain
from graph.chains.router import router_chain, RouteQuery
from graph.state import GraphState
from langgraph.graph import StateGraph, END
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH, LAST

load_dotenv()


def should_search(state: GraphState) -> str:
    search = state['web_search']
    if search:
        return WEBSEARCH
    return GENERATE


def grade_generation_grounded_in_docs_and_questions(state: GraphState) -> str:
    question = state['question']
    documents = state['documents']
    generation = state['generation']
    res = hallucinations_grader_chain.invoke({
        "documents": documents, "generation": generation
    })
    if res.binary_score.lower() == "yes":
        score = answer_grader_chain.invoke({
            "question": question, "generation": generation
        })
        if score.binary_score == 'yes':
            return 'useful'
        else: 
            return 'not useful'
        
    return 'not supported'

def route_input(state: GraphState) -> str:
    question = state['question']
    path: RouteQuery = router_chain.invoke({"question": question})
    if path.datasource == "vectorstore":
        return RETRIEVE
    return WEBSEARCH

builder = StateGraph(GraphState)
builder.add_node(RETRIEVE, retrieve)
builder.set_conditional_entry_point(route_input, {
    RETRIEVE: RETRIEVE,
    WEBSEARCH: WEBSEARCH
})
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEBSEARCH, web_search)
builder.add_node(GENERATE, generate)

builder.add_conditional_edges(GRADE_DOCUMENTS, should_search, {
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
})
builder.add_conditional_edges(GENERATE, grade_generation_grounded_in_docs_and_questions, {
    'useful': END,
    'not useful': WEBSEARCH,
    'not supported': GENERATE
})
builder.add_edge(WEBSEARCH, GENERATE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)



app = builder.compile()
# merm = app.get_graph().draw_mermaid()
# print(merm)


if __name__=="__main__":
    res = app.invoke({"question": "What are the Harness Design Patterns?"})
    print(res)