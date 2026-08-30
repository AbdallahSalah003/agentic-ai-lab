from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    loader = UnstructuredLoader(
        file_path="/home/abdallah/Learning/langchain_agenticai/gist-rag/mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000000
    )
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} chunks created.")

    embeddings = OpenAIEmbeddings(
        base_url="https://backend.sovereigneg.com/v1",
        api_key=os.getenv("SOVEREIGNEG_API_KEY"),
        model="text-embedding-3-small"
    )
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    

if __name__=="__main__":
    main()