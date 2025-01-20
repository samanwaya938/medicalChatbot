from flask import Flask, render_template, jsonify, request
from src.helper import get_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = get_embedding()
index_name = "medicalchatbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatGroq(temperature=0.4, max_tokens=500)

# Initialize chat history
chat_history = []


# Create prompt templates
base_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Context: {context}")
])

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, base_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
history_stuff_chain = create_stuff_documents_chain(llm, contextualize_prompt)
history_retrieval_chain = create_retrieval_chain(history_aware_retriever, history_stuff_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    
    # Use history-aware retrieval if chat history exists
    if chat_history:
        response = history_retrieval_chain.invoke({
            "input": msg,
            "chat_history": chat_history,
            "context": ""
        })
    else:
        response = rag_chain.invoke({"input": msg})
    
    # Update chat history
    chat_history.extend([
        HumanMessage(content=msg),
        AIMessage(content=response["answer"])
    ])
    
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)