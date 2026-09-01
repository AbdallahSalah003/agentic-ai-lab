import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

llm = ChatOpenAI(
    base_url="https://backend.sovereigneg.com/v1",
    api_key=os.getenv("SOVEREIGNEG_API_KEY"),
    model="gpt-oss-120b",
    temperature=0,
)


generation_chain = prompt | llm | StrOutputParser()

