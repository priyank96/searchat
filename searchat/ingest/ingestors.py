from typing import List

import html2text
import requests
from data import store_manager
from langchain.text_splitter import CharacterTextSplitter


class WebIngestor:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    @staticmethod
    def _extract_webpage_text(url, chunk_size: int = 500, separator: str = '\n'):
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, separator=separator)
        html = requests.get(url).text

        page_text = WebIngestor.h.handle(html)
        return text_splitter.split_text(page_text)

    @staticmethod
    def _ingest_text(text: List[str], sources: List[str]):
        store_manager.update_store(text, sources)

    @staticmethod
    def ingest_webpage(url):
        text = WebIngestor._extract_webpage_text(url)
        sources = [0]*len(text)
        for i in range(len(text)):
            sources[i] = {'url': url}
        WebIngestor._ingest_text(text, sources)
        return '<br>'.join(text)
