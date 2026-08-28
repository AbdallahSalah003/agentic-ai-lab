from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Scheme for a source used by an agent"""

    source: str = Field(description="A URL to the source")


class AgentResponse(BaseModel):
    """Scheme for the agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the user's query")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used by the agent to answer the user's query",
    )


MODEL = "gemini-2.5-flash-lite"


llm = ChatGoogleGenerativeAI(model=MODEL)
tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-agenticai!")
    response = agent.invoke(
        input={
            "messages": [
                HumanMessage(
                    content="List 3 job postings for an entry level AI Engineer, in Egypt."
                )
            ]
        }
    )
    print(response)


if __name__ == "__main__":
    main()
