from data.Flan import Flan
from data import store_manager
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class ChatBot:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    faiss_index = store_manager.get_store()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(Flan(), faiss_index.as_retriever(), memory=memory)

    @staticmethod
    def chat(query):
        result = ChatBot.qa({"question": query})
        return result["answer"]