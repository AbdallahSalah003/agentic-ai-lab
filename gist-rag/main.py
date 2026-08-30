import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents.base import Document
from typing import List
from operator import itemgetter
load_dotenv()

base_url="https://backend.sovereigneg.com/v1"
api_key=os.getenv("SOVEREIGNEG_API_KEY")

embeddings = OpenAIEmbeddings(
        base_url=base_url,
        api_key=api_key,
        model="text-embedding-3-small"
    )
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model="gpt-oss-120b"
)
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.getenv("INDEX_NAME")
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""  
)

def format_docs(documents: List[Document]) -> str:
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in documents)


def retrievel_chain_without_lcel(query: str):
    """
    Simple retrievel chain without LCEL.
    Limitations:
    1. Manual step-by-step execution.
    2. No built-in streaming support.
    3. No async support without additional code
    4. Harder to compose with other chains.
    5. More verbose and error-brone.
    """
    relevant_docs = retriever.invoke(input=query)
    context = format_docs(relevant_docs)
    messages = prompt_template.from_messages(context=context, question=query)
    respone = llm.invoke(messages)
    return respone.content

def retrievel_chain_with_lcel():
    """
    Create a retrievel Chain.
    Retruns a chain that can be invoked with {"question": "..."}
    Advantages:
    1. Declerative and composable: easy to chain operations with pipe operator
    2. Built-in streaming: chain.stream() out-of-the-box
    3. Built-in async: chain.ainvoke() & chain.astream() 
    4. Batch processing: chain.batch()
    5. Type safety: better integration with LangChain type system
    6. Less code more concise and readable
    7. Reusability: chain can be saved, shared, or used with other chains
    8. Better debuggind: Observability
    """
    chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") |retriever | format_docs
        )
        | prompt_template 
        | llm 
        | StrOutputParser()
    )
    # format_doc is not a Runnable, but LangChain converts python methods
    # to a RunnableLambda so we could use them in chains. RunnableLambda(format_doc)
    return chain  
     


def main():
    question = "What is Pinecone in Machine Learning?"
    # no_rag_answer = llm.invoke([HumanMessage(content=question)])
    # print("LLM Invocation No RAG:\n")
    # print(no_rag_answer.content)
    # print()
    chain = retrievel_chain_with_lcel()
    response = chain.invoke({"question": question})
    print(response)

if __name__=="__main__":
    main()