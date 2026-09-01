import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    base_url="https://backend.sovereigneg.com/v1",
    api_key=os.getenv("SOVEREIGNEG_API_KEY"),
    model="gpt-oss-120b",
    temperature=0,
)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="documents are relevant to the question 'yes' or 'no'.")


structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question.\n
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n
Give a binary score 'yes' or 'no' score to indicate wether a document is relevant to the question or not.
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    (
        "human",
        "Retrieved Document:\n\n{document}\n\nUser Question:\n{question}"
    ),
])

retriever_grader = grade_prompt | structured_llm_grader



