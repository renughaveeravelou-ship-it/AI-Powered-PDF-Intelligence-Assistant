from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
import json

class GraphState(TypedDict):
    input: str
    chat_history: list
    context: str
    output: str
    retries: int

def build_graph(retriever, llm):
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
        # Feature 4: Citations
        context = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}, Page: {d.metadata.get('page', 'Unknown')}]\n{d.page_content}" for d in docs])
        return {"context": context}

    def web_searcher_node(state: GraphState):
        results = web_search.invoke(state['input'])
        return {"context": f"[Source: Web Search]\n{results}"}

    def synthesizer_node(state: GraphState):
        context = state.get("context", "No external context retrieved.")
        prompt = f"""You are an AI assistant. Answer the user's question based on the context. 
        If you use the context, you MUST cite it at the end of your sentences using the provided [Source, Page] format.
        
        Context:
        {context}
        
        Question: {state['input']}
        Answer:"""
        
        res = llm.invoke(prompt)
        response = res.content if hasattr(res, 'content') else res
        return {"output": response}
        
    def hallucination_evaluator_node(state: GraphState):
        if state.get("retries", 0) >= 2:
            return "end"
            
        context = state.get("context", "")
        if not context or context == "No external context retrieved.":
            return "end" # No context to hallucinate against
            
        prompt = f"""Does the following output hallucinate or make up facts not present in the context?
        Context: {context}
        Output: {state['output']}
        Reply strictly with 'yes' or 'no'."""
        
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
    
    workflow.add_edge("document_retriever", "synthesizer")
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
    
    return workflow.compile()

class AgentExecutorWrapper:
    def __init__(self, graph):
        self.graph = graph
        
    def invoke(self, inputs, config=None):
        state = {
            "input": inputs["input"],
            "chat_history": [],
            "context": "",
            "output": "",
            "retries": 0
        }
        # In LangGraph 0.1.0+, invoke returns the final state
        result = self.graph.invoke(state, config=config)
        return {"output": result["output"]}
