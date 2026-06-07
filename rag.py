# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.

# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None
ef = None

def initialize_components():
    global llm, ef
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)
    if ef is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

def process_urls(urls):
    global vector_store
    yield "Initializing Components"
    initialize_components()
    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)
    yield "Add chunks to vector database...✅"
    vector_store = FAISS.from_documents(docs, ef)
    vector_store.save_local(str(VECTORSTORE_DIR))
    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = ", ".join(set([doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")]))
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content, sources