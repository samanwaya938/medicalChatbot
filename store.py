from src.helper import pdf_loader, text_splitter, get_embedding
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

documents = pdf_loader(data="Data/")
text_chunks = text_splitter(documents)
embedding = get_embedding()



pc = pinecone(PINECONE_API_KEY)
index_name = "medicalchatbot"
pc.create_index(name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))

docserach = PineconeVectorStore.from_documents(documents=text_chunks, embedding=embedding, index_name="medicalchatbot")

