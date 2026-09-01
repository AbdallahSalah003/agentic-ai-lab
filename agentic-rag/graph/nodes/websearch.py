from typing import Any, Dict
from langchain_core.documents.base import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    question = state['question']
    documents = state["documents"] if "documents" in state else None
    results = web_search_tool.invoke({"query": question})["results"]
    join_results = "\n".join(
        [result['content'] for result in results]
    )
    web_search_doc = Document(page_content=join_results)
    if documents is None:
        documents = [web_search_doc]
    else:
        documents.append(web_search_doc)

    return {"question": question, "documents": documents}
        
