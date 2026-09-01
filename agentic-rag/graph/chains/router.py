import os
from typing import Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv



load_dotenv()

llm = ChatOpenAI(
    base_url="https://backend.sovereigneg.com/v1",
    api_key=os.getenv("SOVEREIGNEG_API_KEY"),
    model="gpt-oss-120b",
    temperature=0,
)


class RouteQuery(BaseModel):
    """Route the user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ... ,
        description="Given a user question choose to route it to web search or vectorstore.",
    )



llm_with_structured_output = llm.with_structured_output(RouteQuery)


system = """You are an expert of routing a user question to either a web search or vectorstore. \n
The vectorstore contains documents related to the following: \n
1. Harness Engineering for Self-Improvement 
2. LLM Powered Autonomous Agents 
3. Prompt Engineering \n
Use vectorstore on question related to these topics. For all else, use web search.
"""

prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

router_chain: RunnableSequence = prompt | llm_with_structured_output