from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import RetrievalQA
import textwrap
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

import pdfplumber
import pytesseract
from langchain_core.documents import Document

def get_pdf_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(layout=True) or ""
            
            if len(text.strip()) < 50:
                try:
                    im = page.to_image(resolution=300)
                    ocr_text = pytesseract.image_to_string(im.original)
                    if ocr_text.strip():
                        text += "\n[OCR Extracted Text]\n" + ocr_text
                except Exception as e:
                    pass
                    
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": i + 1}
                ))
    return pages

def initialize_embeddings():
    embeddings = OllamaEmbeddings(model="llama2")
    return embeddings

from langchain_community.chat_models import ChatOpenAI

def initialize_llm(model_type="Ollama (Llama2)"):
    if model_type == "OpenAI (GPT-4)":
        return ChatOpenAI(model="gpt-4", temperature=0)
    return OllamaLLM(model="llama2")

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

def get_ensemble_retriever(pages):
    embeddings = initialize_embeddings()
    vector_db = Chroma.from_documents(pages, embeddings)
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(pages)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6]
    )
    
    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    return compression_retriever

from typing import TypedDict
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    input: str
    chat_history: list
    context: str
    output: str
    retries: int
    research_mode: bool
    domain_persona: str

class AgentExecutorWrapper:
    def __init__(self, graph, message_history):
        self.graph = graph
        self.message_history = message_history
        
    def invoke(self, inputs, config=None):
        state = {
            "input": inputs["input"],
            "chat_history": self.message_history.messages[-5:] if self.message_history else [],
            "context": "",
            "output": "",
            "retries": 0,
            "research_mode": inputs.get("research_mode", False),
            "domain_persona": inputs.get("domain_persona", "General AI Assistant")
        }
        result = self.graph.invoke(state, config=config)
        output = result.get("output", "")
        
        import re
        confidence = "N/A"
        match = re.search(r'\[CONFIDENCE:\s*([^\]]+)\]', output, re.IGNORECASE)
        if match:
            confidence = match.group(1)
            output = re.sub(r'\[CONFIDENCE:\s*[^\]]+\]', '', output, flags=re.IGNORECASE).strip()
            
        if self.message_history:
            self.message_history.add_user_message(inputs["input"])
            self.message_history.add_ai_message(output)
        return {"output": output, "confidence": confidence}

def get_agent_executor(retriever, llm, session_id="default"):
    message_history = SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///memory.db"
    )
    web_search = DuckDuckGoSearchRun()
    
    def router(state: GraphState):
        prompt = f"Given the user question, decide to search 'documents', search 'web', or 'generate' directly. Question: {state['input']}\nReply with just the word: documents, web, or generate."
        res = llm.invoke(prompt)
        decision = (res.content if hasattr(res, 'content') else res).strip().lower()
        if 'doc' in decision:
            return "document_retriever"
        elif 'web' in decision:
            return "web_searcher"
        else:
            return "synthesizer"

    def document_retriever_node(state: GraphState):
        docs = retriever.invoke(state['input'])
        context = state.get("context", "") + "\n\n" + "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}, Page: {d.metadata.get('page', 'Unknown')}]\n{d.page_content}" for d in docs])
        return {"context": context}

    def web_searcher_node(state: GraphState):
        results = web_search.invoke(state['input'])
        context = state.get("context", "") + "\n\n" + f"[Source: Web Search]\n{results}"
        return {"context": context}

    def synthesizer_node(state: GraphState):
        context = state.get("context", "No external context retrieved.")
        history = "\n".join([f"{m.type}: {m.content}" for m in state.get('chat_history', [])])
        persona = state.get("domain_persona", "General AI Assistant")
        
        prompt = f"You are a {persona}. Answer the user's question based on the context. If you use the context, you MUST cite it at the end of your sentences using the provided [Source, Page] format.\nAt the very end of your response, you MUST provide a confidence score representing how well the context answers the question in this exact format: [CONFIDENCE: X/10].\n\nHistory:\n{history}\n\nContext:\n{context}\n\nQuestion: {state['input']}\nAnswer:"
        res = llm.invoke(prompt)
        response = res.content if hasattr(res, 'content') else res
        return {"output": response}
        
    def hallucination_evaluator_node(state: GraphState):
        if state.get("retries", 0) >= 2:
            return "end"
        context = state.get("context", "")
        if not context or context == "No external context retrieved.":
            return "end"
        prompt = f"Does the following output hallucinate or make up facts not present in the context?\nContext: {context}\nOutput: {state['output']}\nReply strictly with 'yes' or 'no'."
        res = llm.invoke(prompt)
        eval_result = (res.content if hasattr(res, 'content') else res).strip().lower()
        if 'yes' in eval_result:
            return "retry"
        return "end"
        
    def retry_node(state: GraphState):
        return {"retries": state.get("retries", 0) + 1, "output": "I need to correct my previous answer. Let me try again."}

    workflow = StateGraph(GraphState)
    workflow.add_node("document_retriever", document_retriever_node)
    workflow.add_node("web_searcher", web_searcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("retry", retry_node)
    
    workflow.set_conditional_entry_point(
        router,
        {
            "document_retriever": "document_retriever",
            "web_searcher": "web_searcher",
            "generate": "synthesizer"
        }
    )
    workflow.add_conditional_edges(
        "document_retriever",
        lambda state: "web_searcher" if state.get("research_mode") else "synthesizer",
        {"web_searcher": "web_searcher", "synthesizer": "synthesizer"}
    )
    workflow.add_edge("web_searcher", "synthesizer")
    workflow.add_conditional_edges(
        "synthesizer",
        hallucination_evaluator_node,
        {
            "end": END,
            "retry": "retry"
        }
    )
    workflow.add_edge("retry", "synthesizer")
    
    graph = workflow.compile()
    return AgentExecutorWrapper(graph, message_history)

def generate_document_summary(retriever, llm):
    """AI Workflow Automation: Generate an automatic summary of the document upon upload."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    summary = qa_chain.invoke("Provide a concise summary of the main topics in this document.")
    return summary['result']

import networkx as nx
from pyvis.network import Network
import tempfile
import json
import re

def generate_knowledge_graph(pages, llm):
    text = " ".join([p.page_content for p in pages[:3]])[:4000]
    prompt = f"Extract up to 10 key entities and their relationships from the text below. Output strictly in this JSON format: [{{\"source\": \"Entity1\", \"target\": \"Entity2\", \"relation\": \"relationship\"}}]\n\nText: {text}"
    try:
        res = llm.invoke(prompt)
        response_text = res.content if hasattr(res, 'content') else res
        json_str = re.search(r'\[.*\]', response_text, re.DOTALL).group()
        data = json.loads(json_str)
        
        G = nx.DiGraph()
        for edge in data:
            G.add_edge(edge['source'], edge['target'], title=edge['relation'])
            
        net = Network(notebook=False, directed=True, bgcolor="#222222", font_color="white", height="400px", width="100%")
        net.from_nx(G)
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.save_graph(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        return None

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(response_text):
    return wrap_text_preserve_newlines(response_text)
