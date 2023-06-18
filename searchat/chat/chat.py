from data import store_manager
from data.Flan import Flan
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory


class ChatBot:
    faiss_index = store_manager.get_store()
    llm = Flan()
    qa = RetrievalQA.from_llm(llm,
                              faiss_index.as_retriever(k=3),
                              prompt="Answer based on context:\n\n{context}\n{question}",
                              )

    @staticmethod
    def chat(query):
        result = ChatBot.qa({"question": query})
        return result["answer"]
