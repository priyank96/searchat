from typing import List

import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter


class WebIngestor:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    @staticmethod
    def _extract_webpage_text(url, chunk_size: int = 1500, separator: str = '\n'):
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, separator=separator)
        html = requests.get(url).text
        soup = BeautifulSoup(html, features="html.parser")
        page_text = soup.get_text()
        lines = (line.strip() for line in page_text.splitlines())
        text = '\n\n'.join(line for line in lines if line)
        return text_splitter.split_text(text)

    @staticmethod
    def _ingest_text(text: List[str]):
        ...

    @staticmethod
    def ingest_webpage(url):
        text = WebIngestor._extract_webpage_text(url)
        WebIngestor._ingest_text(text)
        return text
