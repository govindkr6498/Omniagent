from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import logging

class PDFQATool:
    def __init__(self, pdf_path: str, model_name: str = "gpt-4o-mini"):
        self.logger = logging.getLogger("pdf_qa_tool")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.pdf_path = pdf_path
        self.llm = ChatOpenAI(model=model_name)
        self._load_pdf()
        self._setup_vector_store()

    def _load_pdf(self):
        self.logger.info(f"Loading PDF from {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, length_function=len)
        self.docs = splitter.split_documents(raw_docs)
        self.logger.info(f"Loaded and split PDF into {len(self.docs)} chunks.")

    def _setup_vector_store(self):
        self.logger.info("Setting up vector store for PDF Q&A...")
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(self.docs, embedding=embeddings)
        self.logger.info("Vector store setup complete.")

    def get_context(self, query: str) -> str:
        self.logger.info(f"[RAG] Retrieving context for query: {query}")
        docs = self.vector_store.similarity_search(query, k=5)
        for i, doc in enumerate(docs):
            self.logger.info(f"[RAG] Context chunk {i+1}: {doc.page_content[:200]}...")
        context = "\n".join(doc.page_content for doc in docs)
        self.logger.info(f"[RAG] Combined context: {context[:500]}...")
        return context

    def answer(self, message: str, conversation_history: List[str], lead_info: Dict[str, str], lead_state: str) -> str:
        self.logger.info(f"[RAG] Answering message: {message}")
        context = self.get_context(message)
        self.logger.info(f"[RAG] Context used for answer: {context[:500]}...")
        if not context.strip():
            self.logger.warning("[RAG] No relevant context found for query.")
            return "Sorry, I can only answer questions related to Emaar Properties, meetings, or our services. Please ask something related."
        recent = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        self.logger.info(f"[RAG] Recent conversation history: {recent}")
        topics_prompt = f"Given these conversation messages, identify the main topic being discussed:\n{chr(10).join(recent)}\nReturn ONLY the topic being discussed, nothing else."
        topic_response = self.llm.invoke(topics_prompt)
        current_topic = topic_response.content
        self.logger.info(f"[RAG] LLM topic detected: {current_topic}")
        system_context = f"Current topic: {current_topic}\nProduct info: {context}\nLead info: {lead_info if lead_info else 'None'}\nLead state: {lead_state}"
        prompt = f"""
You are a friendly sales assistant for Emaar.
You must only answer questions related to Emaar properties and emaar FAQs.
after schedule meeting response is "Please give me below information!...".
If the user's question is not related, politely respond: 'Sorry, I can only answer questions related to Emaar Properties, meetings, or our services. Please ask something related.'
Never answer general knowledge or unrelated questions.

System Context:
{system_context}
Human: {message}
Assistant: Be direct and natural, maintain the conversation flow about {current_topic} if relevant.
"""
        self.logger.info(f"[RAG] LLM prompt: {prompt[:500]}...")
        response = self.llm.invoke(prompt)
        self.logger.info(f"[RAG] LLM response: {response.content[:500]}...")
        return response.content
