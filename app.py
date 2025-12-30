from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_embedding_model
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq

app = Flask(__name__)
load_dotenv()

# ---- ENV ----
os.environ["PINECONE_API_KEY"] = os.getenv("vector_db_api_key")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---- LLM ----
chatModel = ChatGroq(model="openai/gpt-oss-120b")

# ---- Embeddings ----
embeddings = download_embedding_model()

# ---- Vector Store ----
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="medibot"
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---- Prompt ----
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# ---- RAG Chain (LCEL) ----
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | chatModel
    | StrOutputParser()
)

# ---- Routes ----
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke(msg)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
