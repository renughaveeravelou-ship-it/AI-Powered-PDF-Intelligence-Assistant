import streamlit as st
import os
import tempfile
import uuid
import io
import base64
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from gtts import gTTS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from utils import get_pdf_pages, get_ensemble_retriever, \
                get_agent_executor, process_llm_response, initialize_llm, generate_document_summary, generate_knowledge_graph

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def text_to_speech(text):
    # Only synthesize the first 500 chars to avoid long blocking calls
    tts = gTTS(text=text[:500], lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        return tmp_file.name

def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        return ""

def main():
    st.set_page_config(
        page_title="PDF Intelligence Chatbot",
        layout="wide"
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19 !important;
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    
    /* Sidebar text visibility overrides */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 550 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {
        color: #cbd5e1 !important;
    }

    /* Style the text in Selectbox dropdown selected value */
    [data-testid="stSidebar"] div[data-baseweb="select"] div {
        color: #f1f5f9 !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] svg {
        fill: #cbd5e1 !important;
    }
    
    /* Popover/listbox dropdown values */
    ul[role="listbox"], [data-testid="stSelectbox"] ul {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    
    ul[role="listbox"] li, [data-testid="stSelectbox"] li {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
    }
    
    ul[role="listbox"] li:hover, [data-testid="stSelectbox"] li:hover {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }

    /* Toggle labels and text elements inside checkbox */
    .stCheckbox label p, [data-testid="stCheckbox"] label p {
        color: #cbd5e1 !important;
    }

    /* Sidebar divider line */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Premium Buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        height: auto !important;
        font-weight: 500 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2), 0 2px 4px -1px rgba(99, 102, 241, 0.1) !important;
        text-transform: none !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3) !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton>button:active {
        transform: translateY(1px) !important;
    }
    
    /* Customize Streamlit Input box */
    div[data-baseweb="input"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
    }
    
    input {
        color: #f1f5f9 !important;
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #0f172a !important;
        border: none !important;
        padding: 0 !important;
    }
    
    [data-testid="stFileUploader"] > section {
        background-color: #1e293b !important;
        border: 2px dashed #334155 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: #cbd5e1 !important;
        transition: border-color 0.2s ease !important;
    }

    [data-testid="stFileUploader"] > section:hover {
        border-color: #4f46e5 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
        padding: 0.25rem 0.75rem !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #475569 !important;
    }

    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] span,
    [data-testid="stFileUploader"] small {
        color: #cbd5e1 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    .streamlit-expanderContent {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }

    /* Custom Chat Message aesthetics */
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 16px !important;
        margin-bottom: 12px !important;
        padding: 1.25rem !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="flex items-center gap-4 py-6 mb-8 border-b border-slate-800/80">
        <div class="w-12 h-12 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-2xl shadow-lg shadow-indigo-500/20">
            📚
        </div>
        <div>
            <h1 class="text-3xl font-bold text-slate-100 tracking-tight">PDF Intelligence Chatbot</h1>
            <p class="text-sm text-indigo-400 font-medium">Powered by LangChain Agents & Llama2 (Multimodal & Voice Enabled)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("LLM Settings")
        st.session_state['selected_llm'] = st.selectbox("Select Model", ["Ollama (Llama2)", "OpenAI (GPT-4)"])
        st.divider()

        st.header("Upload & Process PDFs")
        pdf_files = st.file_uploader("Upload your documents", type=["pdf"], accept_multiple_files=True)
        
        if pdf_files:
            all_pages = []
            for pdf_file in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_path = tmp_file.name
                    pages = get_pdf_pages(tmp_path)
                    for page in pages:
                        page.metadata['source'] = pdf_file.name
                    all_pages.extend(pages)

            if st.button('Process Documents'):
                with st.status("Analyzing documents...", expanded=True) as status:
                    st.write("Extracting text & formulas (PyMuPDF)...")
                    st.write("Generating embeddings & Hybrid Index...")
                    retriever = get_ensemble_retriever(all_pages)
                    st.write(f"Initializing {st.session_state['selected_llm']} model...")
                    llm = initialize_llm(model_type=st.session_state['selected_llm'])
                    
                    st.write("Setting up Agent & Memory...")
                    st.session_state['agent'] = get_agent_executor(retriever, llm, session_id=st.session_state.session_id)
                    
                    st.write("Executing Auto-Summary Workflow...")
                    doc_summary = generate_document_summary(retriever, llm)
                    st.session_state['doc_summary'] = doc_summary
                    
                    st.write("Extracting Analytics & Generating Knowledge Graph...")
                    kg_html_path = generate_knowledge_graph(all_pages, llm)
                    st.session_state['kg_path'] = kg_html_path
                    st.session_state['analytics'] = {
                        "Total Pages": len(all_pages),
                        "Total Characters": sum(len(p.page_content) for p in all_pages),
                        "Reading Time": f"{sum(len(p.page_content.split()) for p in all_pages) // 200} mins"
                    }
                    
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    st.success("Documents ready for chat!")
                    
        if "doc_summary" in st.session_state:
            st.markdown('<div class="text-[11px] font-bold text-indigo-400 uppercase tracking-widest mb-2 mt-6">Document Summary</div>', unsafe_allow_html=True)
            with st.expander("Show Auto-Summary", expanded=False):
                st.write(st.session_state['doc_summary'])
                
        if "analytics" in st.session_state:
            st.markdown('<div class="text-[11px] font-bold text-indigo-400 uppercase tracking-widest mb-2 mt-6">Document Analytics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="grid grid-cols-3 gap-2 mb-4">
                <div class="p-3 bg-gray-800 bg-opacity-40 border border-gray-700 border-opacity-50 rounded-xl text-center">
                    <div class="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-0.5">Pages</div>
                    <div class="text-lg font-bold text-indigo-400">{st.session_state['analytics']["Total Pages"]}</div>
                </div>
                <div class="p-3 bg-gray-800 bg-opacity-40 border border-gray-700 border-opacity-50 rounded-xl text-center">
                    <div class="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-0.5">Chars</div>
                    <div class="text-lg font-bold text-purple-400">{st.session_state['analytics']["Total Characters"]}</div>
                </div>
                <div class="p-3 bg-gray-800 bg-opacity-40 border border-gray-700 border-opacity-50 rounded-xl text-center">
                    <div class="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-0.5">Read Time</div>
                    <div class="text-xs font-bold text-pink-400 mt-1">{st.session_state['analytics']["Reading Time"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if "kg_path" in st.session_state and st.session_state['kg_path']:
            st.markdown('<div class="text-[11px] font-bold text-indigo-400 uppercase tracking-widest mb-2 mt-6">Knowledge Graph</div>', unsafe_allow_html=True)
            with open(st.session_state['kg_path'], 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=400)
        
        st.markdown('<div class="text-[11px] font-bold text-indigo-400 uppercase tracking-widest mb-2 mt-6">Mode Selection</div>', unsafe_allow_html=True)
        st.session_state['domain_persona'] = st.selectbox("Domain Persona", ["General AI Assistant", "Senior Legal Counsel", "Expert Medical Researcher", "Financial Analyst"])
        st.session_state['coding_mode'] = st.toggle("AI Coding Assistant Mode", value=False)
        st.session_state['research_mode'] = st.toggle("Deep Research Mode", value=False)
        st.session_state['auto_voice'] = st.toggle("Auto-Play Voice Responses", value=False)

        st.markdown('<div class="text-[11px] font-bold text-indigo-400 uppercase tracking-widest mb-2 mt-6">AI Report & Collaboration</div>', unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Generate Report"):
                if "messages" in st.session_state and st.session_state.messages:
                    report = "# AI Chatbot Research Report\n\n"
                    if "doc_summary" in st.session_state:
                        report += f"## Document Summary\n{st.session_state['doc_summary']}\n\n"
                    report += "## Conversation Log\n\n"
                    for m in st.session_state.messages:
                        report += f"**{m['role'].capitalize()}**: {m['content']}\n\n"
                    st.download_button("Download MD", data=report, file_name="AI_Report.md", mime="text/markdown")
                else:
                    st.warning("Chat history is empty!")
        
        with col_btn2:
            import json
            if "messages" in st.session_state and st.session_state.messages:
                state_json = json.dumps(st.session_state.messages)
                st.download_button("Export State", data=state_json, file_name="session_state.json", mime="application/json")
                
        uploaded_state = st.file_uploader("Import Session State", type=["json"])
        if uploaded_state:
            try:
                st.session_state.messages = json.loads(uploaded_state.getvalue().decode('utf-8'))
                st.success("Session imported!")
            except:
                st.error("Failed to import.")
                
        st.markdown('<div class="text-[11px] font-bold text-indigo-400 uppercase tracking-widest mb-2 mt-6">AI Workflows</div>', unsafe_allow_html=True)
        col_wf1, col_wf2 = st.columns(2)
        with col_wf1:
            if st.button("Key Entities"):
                st.session_state['macro_prompt'] = "Extract all key entities, dates, and names from the document in a structured list."
        with col_wf2:
            if st.button("Core Clauses"):
                st.session_state['macro_prompt'] = "Identify and summarize the core clauses, constraints, or main arguments from the document."

        st.markdown('<div class="text-[11px] font-bold text-indigo-400 uppercase tracking-widest mb-2 mt-6">Multimodal Vision</div>', unsafe_allow_html=True)
        image_file = st.file_uploader("Upload image for LLM analysis", type=["png", "jpg", "jpeg"])
        if image_file:
            st.image(image_file, use_container_width=True)
            if st.button("Describe Image"):
                encoded_image = base64.b64encode(image_file.getvalue()).decode('utf-8')
                llava = ChatOllama(model="llava")
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image in detail. Focus on the main subjects and context."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                )
                with st.spinner("Analyzing image with LLaVA..."):
                    try:
                        desc = llava.invoke([message]).content
                        st.info(desc)
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        st.session_state.messages.append({"role": "assistant", "content": f"**Image Analysis:**\n{desc}"})
                    except Exception as e:
                        st.error(f"Failed to run vision model. Is `ollama run llava` installed? Error: {e}")

        st.markdown('<div class="h-6"></div>', unsafe_allow_html=True)
        st.info("Ensure Ollama is running with `llama2` and `llava` models.")

    # Main Chat Area
    if "agent" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            text_prompt = st.chat_input("Ask anything about the documents or web...")
        with col2:
            st.markdown('<div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">Voice Input</div>', unsafe_allow_html=True)
            audio = mic_recorder(start_prompt="Record", stop_prompt="Stop", key='recorder')
            
        prompt = st.session_state.pop('macro_prompt', None) or text_prompt
        if audio and "audio_bytes" in audio and audio["audio_bytes"]:
            with st.spinner("Transcribing..."):
                prompt = transcribe_audio(audio["audio_bytes"])

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                stream_handler = StreamHandler(message_placeholder)
                with st.spinner("Agent is thinking (searching docs/web)..."):
                    try:
                        enhanced_prompt = prompt
                        if st.session_state.get('coding_mode', False):
                            enhanced_prompt = f"You are an expert AI Coding Assistant. Explain the concepts using code blocks with clear syntax highlighting, based strictly on the context. User query: {prompt}"
                            
                        response = st.session_state['agent'].invoke(
                            {
                                "input": enhanced_prompt, 
                                "research_mode": st.session_state.get('research_mode', False),
                                "domain_persona": st.session_state.get('domain_persona', "General AI Assistant")
                            },
                            config={"callbacks": [stream_handler]}
                        )
                        final_output = process_llm_response(response["output"])
                        confidence = response.get("confidence", "N/A")
                        
                        confidence_badge = ""
                        if confidence != "N/A":
                            confidence_badge = f"""
                            <div class="mt-4 flex items-center gap-2">
                                <span class="text-xs font-semibold text-gray-400">Retrieval Confidence Score:</span>
                                <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500 bg-opacity-20 text-emerald-400 border border-emerald-500 border-opacity-30">
                                    {confidence}
                                </span>
                            </div>
                            """
                        
                        message_placeholder.markdown(f"{final_output}\n\n{confidence_badge}", unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"{final_output}\n\n{confidence_badge}"
                        })
                        
                        # Generate Audio
                        if st.session_state.get('auto_voice', False):
                            audio_path = text_to_speech(final_output)
                            st.audio(audio_path)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.image("figures/banner.png", use_container_width=True)
        st.markdown("""<div class="text-center py-6">
            <h1 class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
                Welcome to PDF Intelligence Chatbot
            </h1>
            <p class="text-gray-400 mt-2 text-base">
                Unlock the power of your documents with advanced LangChain agents, hybrid search indexing, and real-time vision analytics.
            </p>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 my-8">
            <div class="p-6 bg-gray-800 bg-opacity-40 border border-gray-700 border-opacity-50 rounded-2xl hover:border-indigo-500 hover:border-opacity-50 transition-all duration-300">
                <div class="w-12 h-12 rounded-xl bg-indigo-900 bg-opacity-20 text-indigo-400 flex items-center justify-center text-2xl mb-4 font-bold">
                    
                </div>
                <h3 class="text-lg font-semibold text-gray-100 mb-2">Hybrid Ensemble Retrieval</h3>
                <p class="text-gray-400 text-sm leading-relaxed">
                    Combines BM25 lexical search with vector dense retrieval and FlashRank reranking for hyper-accurate document context.
                </p>
            </div>
            <div class="p-6 bg-gray-800 bg-opacity-40 border border-gray-700 border-opacity-50 rounded-2xl hover:border-purple-500 hover:border-opacity-50 transition-all duration-300">
                <div class="w-12 h-12 rounded-xl bg-purple-900 bg-opacity-20 text-purple-400 flex items-center justify-center text-2xl mb-4 font-bold">
                    
                </div>
                <h3 class="text-lg font-semibold text-gray-100 mb-2">Voice &amp; Vision Enabled</h3>
                <p class="text-gray-400 text-sm leading-relaxed">
                    Transcribe spoken prompts in real time, auto-play answers in audio, and analyze diagrams or document images with LLaVA.
                </p>
            </div>
            <div class="p-6 bg-gray-800 bg-opacity-40 border border-gray-700 border-opacity-50 rounded-2xl hover:border-pink-500 hover:border-opacity-50 transition-all duration-300">
                <div class="w-12 h-12 rounded-xl bg-pink-900 bg-opacity-20 text-pink-400 flex items-center justify-center text-2xl mb-4 font-bold">
                    
                </div>
                <h3 class="text-lg font-semibold text-gray-100 mb-2">Automated Workflows</h3>
                <p class="text-gray-400 text-sm leading-relaxed">
                    Instantly generate document summaries, construct interactive entity knowledge graphs, and export structured PDF research reports.
                </p>
            </div>
        </div>
        <div class="flex items-center gap-4 p-4 rounded-xl bg-indigo-900 bg-opacity-25 border border-indigo-800 border-opacity-40 shadow-inner mb-8">
            <span class="text-2xl"></span>
            <div class="text-sm text-indigo-200">
                <strong>Getting Started:</strong> Upload your PDF document(s) in the sidebar on the left, then click <strong>"Process Documents"</strong> to begin chatting.
            </div>
        </div>""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
