import os
from dotenv import load_dotenv
from src.helper import load_pdf, filter_to_minimal_doc, text_split, download_embedding_model
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

load_dotenv()

chatModel = ChatGroq(model="openai/gpt-oss-120b")

os.environ["PINECONE_API_KEY"] = os.getenv("vector_db_api_key")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

extracted_docs = load_pdf("data/medical.pdf")
filtered_docs = filter_to_minimal_doc(extracted_docs)
docs_chunk = text_split(filtered_docs)

embeddings = download_embedding_model()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "medibot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  
        )
    )

index = pc.Index(index_name)

# Store the embeddings
docsearch = PineconeVectorStore.from_documents(
    documents=docs_chunk,
    embedding=embeddings,
    index_name=index_name
)