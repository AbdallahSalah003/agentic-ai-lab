from typing import Any, Dict
from graph.state import GraphState
from graph.chains.retriever_grader import retriever_grader


def grade_documents(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]
    web_search = False
    filtered_docs = []
    for doc in documents:
        score = retriever_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(doc)
        else:
            web_search=True

    return {"documents": filtered_docs, "question": question, "web_search": web_search}