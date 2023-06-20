import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

if not os.path.isfile('faiss/index.faiss'):
    faiss_index = FAISS.from_texts([""], hf)
else:
    faiss_index = FAISS.load_local('faiss', hf)


def save_store():
    ...


def get_store():
    return faiss_index


def update_store(texts, sources):
    faiss_index.add_texts(texts=texts, metadatas=sources)
