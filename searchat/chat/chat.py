from data import store_manager
from data import Flan, BASIC_DOCUMENT_QA_PROMPT
from langchain.chains import RetrievalQA


class ChatBot:
    faiss_index = store_manager.get_store()
    llm = Flan()
    qa = RetrievalQA.from_llm(llm,
                              faiss_index.as_retriever(k=3),
                              prompt=BASIC_DOCUMENT_QA_PROMPT
                              )

    @staticmethod
    def chat(query):
        result = ChatBot.qa({"question": query})
        return result["answer"]
