import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(
    base_url="https://backend.sovereigneg.com/v1",
    api_key=os.getenv("SOVEREIGNEG_API_KEY"),
    model="gpt-oss-120b",
    temperature=0,
)


class GradeHallucinations(BaseModel):
    """Binary score for hallucinations present in generated answer."""
    binary_score: str = Field(description="Answer is grounded in facts, 'yes' or 'no'")


llm_with_structured_output = llm.with_structured_output(GradeHallucinations)


system = """You are a grader assessing wether an LLM generated answer is grounded / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'yes' means the answer is grounded in / supported by the set of facts.
"""

prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucinations_grader_chain: RunnableSequence = prompt | llm_with_structured_output