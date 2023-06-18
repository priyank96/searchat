from langchain.prompts import PromptTemplate

BASIC_DOCUMENT_QA_PROMPT = PromptTemplate(
    input_variables=['context', 'question'],
    template='Answer based on context:\n\n{context}\n{question}'
)
