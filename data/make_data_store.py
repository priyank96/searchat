from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

loader = CSVLoader(file_path='courses.csv', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


model_name = "sentence-transformers/all-mpnet-base-v2"
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)
faiss_index = FAISS.from_documents(docs, hf)
faiss_index.save_local("courses_docs")
query = "What course related to product management"
docs = faiss_index.similarity_search(query)

for i in range(3):
    print(docs[i].page_content)
