import time

def test_import(module_name):
    t0 = time.time()
    print(f"Importing {module_name}...", end="", flush=True)
    try:
        __import__(module_name)
        print(f" SUCCESS ({time.time() - t0:.2f}s)", flush=True)
    except Exception as e:
        print(f" FAILED ({time.time() - t0:.2f}s): {e}", flush=True)

modules = [
    "langchain_ollama",
    "langchain_community.document_loaders",
    "langchain_community.vectorstores",
    "langchain_classic.chains",
    "textwrap",
    "langchain_classic.retrievers",
    "langchain_community.retrievers",
    "langchain_core.tools.retriever",
    "langchain_community.tools",
    "langchain_classic.agents",
    "langchain_classic.memory",
    "langchain_community.chat_message_histories",
    "flashrank",
    "langgraph.graph"
]

for m in modules:
    test_import(m)


    # hlo
