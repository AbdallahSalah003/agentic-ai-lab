import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()


urls = [
    "https://lilianweng.github.io/posts/2026-07-04-harness/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
]


docs = [UnstructuredLoader(web_url=url, chunking_strategy='basic', max_characters=1000_000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(docs_list)

for chunk in chunks:
    chunk.metadata ={
        "source": chunk.metadata.get("url", "UNKNOWN"),
        "filetype": chunk.metadata.get("filetype", "text/html"),
        "element_id": chunk.metadata.get("element_id", 'N/A')
    }

embeddings = OpenAIEmbeddings(
    base_url="https://backend.sovereigneg.com/v1",
    api_key=os.getenv("SOVEREIGNEG_API_KEY"),
    model="text-embedding-3-small"
)

# vectorstore = PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=os.getenv("INDEX_NAME"))

vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv("INDEX_NAME"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 