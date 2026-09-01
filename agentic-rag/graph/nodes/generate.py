from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState



def generate(state: GraphState) -> Dict[str, Any]:
    documents = state["documents"]
    question = state["question"]
    res = generation_chain.invoke({"question": question, "context": documents})
    return {"question": question, "documents": documents, "generation": res}

