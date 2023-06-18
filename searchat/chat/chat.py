from data import Flan, store_manager, BASIC_DOCUMENT_QA_PROMPT
from langchain.chains import RetrievalQA


class ChatBot:
    faiss_index = store_manager.get_store()
    llm = Flan()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_index.as_retriever(k=3),
        chain_type_kwargs={
            "prompt": BASIC_DOCUMENT_QA_PROMPT
        },
    )

    @staticmethod
    def chat(query):
        result = ChatBot.qa({"query": query})
        return result["result"]
