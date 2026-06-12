from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI

# Unsloth + HuggingFace imports
from unsloth import FastLanguageModel
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms.ollama import Ollama

# Correct imports for 0.4.x
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import torch
import re

# Global chat history list
chat_history = []

SOP_QUERIES = [
    "Does the document mention safety guidelines?",
    "Is there a section about quality assurance?",
    "Does it include compliance with ISO standards?",
    "Are emergency procedures described?",
]

# ✅ Load fine‑tuned model once at startup
print("🔄 Loading fine‑tuned model...")
ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
    model_name="C:\\Users\\hp\\Documents\\Multi-fileChatBot\\backend\\my_llm",
    max_seq_length=2048,
    dtype=torch.bfloat16,    # ← Explicit, no fallback ambiguity
    load_in_4bit=True,       # Essential for 8GB VRAM
    device_map="auto",     # ← Explicit CUDA device, not "auto"
)

try:
    ft_model = FastLanguageModel.get_peft_model(ft_model)
    print("✅ LoRA adapters applied")
except Exception as e:
    print("ℹ️ No adapters found or already applied:", e)

ft_pipeline = pipeline(
    "text-generation",
    model=ft_model,
    tokenizer=ft_tokenizer,
    max_new_tokens=512,
    temperature=0.7,
)
ft_llm = HuggingFacePipeline(pipeline=ft_pipeline)

qwen_llm = OpenAI(
    model="qwen3.5-9b-sushi-coder-rl",   # use the exact LM Studio model name
    openai_api_base="http://localhost:1234/v1",  # LM Studio server
    openai_api_key="lm-studio",   # dummy key, LM Studio ignores it
    max_tokens=512,
    temperature=0.7
)

def get_chain(db_path: str, use_finetuned: bool = False, use_llava: bool = True, use_base_llama: bool = False, use_qwen: bool = False):
    print(f"Loading database from: {db_path}")

    embeddings = HuggingFaceEmbeddings(model_name="C:\\Users\\hp\\Documents\\Multi-fileChatBot\\models\\all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 20})

    # ✅ Choose model dynamically
    if use_finetuned:
        llm = ft_llm
        print("⚡ Using fine‑tuned model")
    elif use_llava:
        llm = Ollama(model="llava:7b")
        print("⚡ Using LLaVA multimodal model")
    elif use_base_llama:
        llm = Ollama(model="mistral:7b")
        print("⚡ Using base Mistral-7b model")
    elif use_qwen:
        llm = qwen_llm
        print("⚡ Using Qwen 3.5-9b-sushi-coder-rl from LM Studio")
        
    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are answering based on multiple PDF pages.
        Check all provided context before answering.
        Combine information from ALL retrieved pages.
        Do not stop after the first match.
        If relevant information is found on more than one page then provide the answer based on all of them.
        If asked to compare between two documents then compare them by making tables side by side.
        Additionally explain in simple, factual terms and give a brief summary of the relevant information from the documents.
        You are answering questions from technical PDFs.
        Use semantically related information from the context.
        A concept may be described using different terminology.
        If relevant information exists, summarize it instead of saying
        the answer is unavailable.
        Only say "not found" when the context is completely unrelated.
        if the query is for two then give resulf on keyword and.
        Context:
        {context}
        Question: {input}
        """
    )

    chain = (
        {
            "context": RunnableLambda(lambda x: retriever.invoke(x["input"])),
            "input": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(query: str, db_path: str, use_finetuned: bool = False, use_llava: bool = True, use_base_llama: bool = False, use_qwen: bool = False):
    global chat_history

    if len(chat_history) > 10:
        chat_history = chat_history[-6:]

    chain, retriever = get_chain(db_path, use_finetuned=use_finetuned, use_llava=use_llava, use_base_llama=use_base_llama)
    answer = chain.invoke({"input": query, "chat_history": chat_history})

    retrieved_documents = retriever.invoke(query)

    chat_history.append(("User", query))
    chat_history.append(("Assistant", answer))

    reference = []
    seen = set()
    for doc in retrieved_documents:
        ref = f"{doc.metadata.get('source')} (page {doc.metadata.get('page')})"
        if ref not in seen:
            reference.append(ref)
            seen.add(ref)

    return answer, reference

def extract_items(text):
    """
    Extracts both numeric values with units and textual specifications.
    Splits on commas, slashes, semicolons, and newlines.
    Returns a set of normalized items.
    """
    if not text:
        return set()
    text = text.lower()
    items = re.split(r"[,\n;/]", text)
    return set([item.strip() for item in items if item.strip()])

def sop_check(queries, ref_db="chroma_db_ref", check_db="chroma_db_check", use_qwen=True):
    results = {}
    for query in queries:
        ref_answer, ref_refs = ask(query, ref_db, use_qwen=use_qwen)
        check_answer, check_refs = ask(query, check_db, use_qwen=use_qwen)

        ref_items = extract_items(ref_answer)
        check_items = extract_items(check_answer)

        # Compare sets
        missing_in_check = ref_items - check_items
        extra_in_check = check_items - ref_items
        matched = not missing_in_check and not extra_in_check and ref_items != set()

        results[query] = {
            "reference_answer": ref_answer,
            "checking_answer": check_answer,
            "reference_items": list(ref_items),
            "checking_items": list(check_items),
            "matched": matched,
            "missing_in_check": list(missing_in_check),
            "extra_in_check": list(extra_in_check),
            "references": {"ref": ref_refs, "check": check_refs}
        }
    return results
