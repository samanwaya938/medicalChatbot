from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def pdf_loader(path):
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls= PyPDFLoader)
    pages = loader.load()
    return pages

def text_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(text)
    return text_chunks

def get_embedding():
  embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  return embedding