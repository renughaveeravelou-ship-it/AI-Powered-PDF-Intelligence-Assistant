#  AI-Powered PDF Intelligence Assistant

An advanced AI-Powered PDF Intelligence Assistant built using <br>
LangChain,
Ollama,
Streamlit, and
OpenAI.

This project allows users to upload PDF documents and interact with them using conversational 


# AI with features 

 PDF Question Answering
 Hybrid AI Retrieval System
 Web Search Integration
 Voice Input & AI Voice Responses
 Multimodal Image Understanding (LLaVA)
 Knowledge Graph Generation
 Automatic Document Summarization
 Domain-Specific AI Personas
 Deep Research Mode
 Session Memory & Export System

Built for advanced AI/ML portfolio projects and real-world intelligent document analysis systems.

# Project Preview
PDF Intelligence Chatbot
Upload PDFs → Process Documents → Chat with AI → Generate Insights

# Features
 -Intelligent PDF Chat
Ask natural language questions from uploaded PDFs.
-Multi-PDF support
-OCR support for scanned PDFs
-Source-aware answers with page citations
-conversational memory


# Hybrid Retrieval Pipeline

This project combines:
-BM25 Retrieval
-Dense Vector Retrieval
-FlashRank Re-ranking
-Ensemble Retrieval Architecture
for highly accurate contextual responses.

# AI Agent Workflow System

Uses LangGraph-based AI workflows:
-Document Retrieval
-Web Search Routing
-Response Synthesis
-Hallucination Detection
-Retry Correction Mechanism

# Web Search Integration

Integrated DuckDuckGo Search for real-time research augmentation.
Perfect for:
-Research papers
-Legal analysis
-Financial reports
-Medical documents

# Voice AI

Supports:
_Speech-to-Text
-Voice-based prompting
-AI-generated audio responses using Google TTS

# Multimodal Vision AI

Upload images and analyze them using:
-LLaVA Vision Model
-Image Captioning
-Visual Understanding


# Knowledge Graph Generation

Automatically extracts:

Entities
Relationships
Concepts

and generates interactive visual knowledge graphs.

# Auto Document Summary

Generate concise summaries instantly after document upload.

# Domain Personas

Switch AI behavior dynamically:

General AI Assistant
Senior Legal Counsel
Medical Research Expert
Financial Analyst

# Deep Research Mode

Combines:

Document Retrieval
Web Search
Context Synthesis

for advanced AI research workflows.

# Tech Stack


Frontend
Streamlit
TailwindCSS
Backend
Python
LangChain
LangGraph
AI/LLM
Ollama (Llama2)
OpenAI GPT-4
LLaVA
Vector Database
ChromaDB
Retrieval & Ranking
BM25
FlashRank
Ensemble Retrieval
OCR
Tesseract OCR
PyMuPDF
pdfplumber
Database
SQLite Memory Storage

# Project Structure

├── app.py                 # Main Streamlit application
├── main.py                # CLI-based PDF chatbot
├── utils.py               # Core AI pipeline utilities
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker setup
├── docker-compose.yml     # Multi-container deployment
├── memory.db              # Chat memory database
├── figures/               # UI assets/images
└── pdfs/                  # Sample PDFs


 # Installation
1️ Clone Repository
git clone https://github.com/your-username/pdf-intelligence-chatbot.git
cd pdf-intelligence-chatbot

2️ Create Virtual Environment
python -m venv venv
Activate Environment
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate

3️ Install Dependencies
pip install -r requirements.txt

# Install Ollama

Download and install:

Ollama Official Website

Pull required models:

ollama pull llama2
ollama pull llava

# Run Application
Streamlit App
streamlit run app.py
CLI Mode
python main.py --pdf_file pdfs/paper.pdf

# Docker Setup
Build Docker Container
docker build -t pdf-chatbot .
Run Container
docker run -p 8501:8501 pdf-chatbot

# Example Use Cases
Research Paper Assistant
Legal Contract Analysis
Medical Report Understanding
Financial Report Insights
AI Knowledge Management
Academic Study Assistant
Enterprise Document Intelligence

# Advanced AI Concepts Used

Retrieval-Augmented Generation (RAG)
Agentic AI Workflows
Hybrid Search Systems
Context Compression
AI Memory Systems
Multimodal AI
Hallucination Detection
Conversational AI

# Important Libraries

Based on your project dependencies:

LangChain
LangGraph
ChromaDB
Streamlit
Ollama
PyMuPDF
FlashRank
SpeechRecognition
TensorFlow
Transformers
OpenCV

# Future Improvements

Authentication System
Multi-user Collaboration
Cloud Deployment
PDF Annotation System
Citation-aware AI Responses
Vector Database Scaling
Fine-tuned Domain Models
Real-time Streaming Responses


# Contributing

-Fork the repository
-Create a feature branch
-Commit changes
-Push to branch
-Open Pull Request

# License

This project is developed for educational and learning purposes only.

# Author
Renugha V