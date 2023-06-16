from data import store_manager
from data.Flan import Flan
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class ChatBot:
    faiss_index = store_manager.get_store()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(Flan(), faiss_index.as_retriever(), memory=memory)

    @staticmethod
    def chat(query):
        result = ChatBot.qa({"question": query})
        return result["answer"]
