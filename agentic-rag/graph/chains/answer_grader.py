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


class GradeAnswer(BaseModel):
    """Binary score for the answer."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


llm_with_structured_output = llm.with_structured_output(GradeAnswer)


system = """You are a grader assessing wether an answer addresses / resolves a question. \n
Give a binary score 'yes' or 'no'. 'yes' means the answer addresses / resolves the question.
"""

answer_prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "Question: \n\n {question} \n\n Answer: {generation}"),
    ]
)

answer_grader_chain: RunnableSequence = answer_prompt | llm_with_structured_output