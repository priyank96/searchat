from data import store_manager
from data.Flan import Flan
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory


class ChatBot:
    faiss_index = store_manager.get_store()

    qa = RetrievalQA.from_llm(Flan(), faiss_index.as_retriever())

    @staticmethod
    def chat(query):
        result = ChatBot.qa({"question": query})
        return result["answer"]
