from data import AutoLM, Flan, store_manager, BASIC_DOCUMENT_QA_PROMPT
from langchain.chains import RetrievalQA


class ChatBot:
    faiss_index = store_manager.get_store()
    llm = AutoLM()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_index.as_retriever(k=3),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": BASIC_DOCUMENT_QA_PROMPT
        },
    )

    @staticmethod
    def chat(query):
        response = ChatBot.qa({"query": query})
        return response["result"].replace('\n', '<br>'), '<pre>' + '<br><hr>'.join(
            [x.page_content + '<pre>' + x.metadata['url'] for x in response["source_documents"]])
