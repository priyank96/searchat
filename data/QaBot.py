from Flan import Flan
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


model_name = "sentence-transformers/all-mpnet-base-v2"
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)
faiss_index = FAISS.load_local('data/courses_docs', hf)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(Flan(), faiss_index.as_retriever(), memory=memory)

query = "When did GDPR become a law?"
result = qa({"question": query})
print(result["answer"])