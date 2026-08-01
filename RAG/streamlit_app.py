"""
NHPT Heritage RAG Assistant — Streamlit App
=============================================
Streamlit front-end for the CV + RAG pipeline built in rag_assistant.ipynb.

Flow:
  1. User uploads a photo of a building.
  2. The Part B ResNet50 model predicts the architectural style.
  3. The prediction (with confidence) is handed to a LangChain RAG chain,
     grounded in knowledge_base/*.txt and served by Llama 3.1 8B Instruct
     via Hugging Face Inference Providers.
  4. The user can keep chatting about the result — multi-turn memory is
     kept per browser session in st.session_state.

Run with:  streamlit run streamlit_app.py

Setup:
  - Put a `.env` file next to this script containing:  HF_TOKEN=hf_xxx
  - Make sure models/best_resnet50.pth and models/model_metadata.json exist
    (from cv_classification.ipynb)
  - knowledge_base/ should contain one .txt file per class (auto-seeded on
    first run if empty)
"""

import os
import json
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ── Page config must be the first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="NHPT Heritage Assistant",
    page_icon="🏛️",
    layout="wide",
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (mirrors RAG_CONFIG in rag_assistant.ipynb)
# ---------------------------------------------------------------------------
RAG_CONFIG = {
    "hf_base_url": "https://router.huggingface.co/v1",
    "llm_model": "meta-llama/Llama-3.1-8B-Instruct:novita",
    "llm_temperature": 0.3,
    "llm_max_tokens": 600,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "retrieval_k": 4,
    "relevance_threshold": 0.3,
    "low_confidence_threshold": 0.6,
    "knowledge_base_dir": "./knowledge_base",
    "chroma_db_dir": "./chroma_db",
    "model_path": "./models/best_resnet50.pth",
    "model_metadata_path": "./models/model_metadata.json",
}

SYSTEM_PROMPT = """You are a knowledgeable heritage guide for the National Heritage Preservation Trust (NHPT).
You help visitors understand the architectural styles and artifacts at historic sites.

RULES:
1. Answer ONLY using the CONTEXT below. Do not rely on outside knowledge.
2. Always name the source document you drew from (e.g. "based on the source on {{style}}...").
3. If the context does not contain enough information, reply exactly:
   "I don't have enough information in my knowledge base to answer that question."
4. Keep answers to 2-4 short paragraphs.
5. If a CV confidence score is given below 60%, explicitly flag the uncertainty and mention
   the next most likely style before answering.

CONTEXT:
{context}
"""


# ---------------------------------------------------------------------------
# Cached resource loaders — run once per server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base & vector store...")
def load_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    import chromadb

    kb_dir = Path(RAG_CONFIG["knowledge_base_dir"])
    if not kb_dir.exists():
        raise FileNotFoundError(
            f"{kb_dir}/ does not exist. Create it and add one .txt file per class."
        )

    kb_documents = []
    for file_path in sorted(kb_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        kb_documents.append(Document(
            page_content=text,
            metadata={"source": file_path.name, "style": file_path.stem},
        ))
    if not kb_documents:
        raise RuntimeError(f"No .txt files found in {kb_dir}/ — add at least one document.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(kb_documents)

    embeddings = HuggingFaceEmbeddings(model_name=RAG_CONFIG["embedding_model"])

    chroma_path = RAG_CONFIG["chroma_db_dir"]
    collection_name = "nhpt_heritage"

    client = chromadb.PersistentClient(path=chroma_path)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
    del client

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
        collection_name=collection_name,
    )
    return vectorstore, len(kb_documents), len(chunks)


@st.cache_resource(show_spinner="Loading heritage classifier model...")
def load_cv_model():
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    metadata_path = Path(RAG_CONFIG["model_metadata_path"])
    model_path = Path(RAG_CONFIG["model_path"])
    if not metadata_path.exists() or not model_path.exists():
        return None  # allow the app to still run in text-only mode

    with open(metadata_path) as f:
        metadata = json.load(f)

    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]
    img_size = metadata["img_size"]
    mean, std = metadata["imagenet_mean"], metadata["imagenet_std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(n):
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n),
        )
        return m

    model = build_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return {
        "model": model,
        "transform": transform,
        "device": device,
        "class_names": class_names,
    }


@st.cache_resource(show_spinner=False)
def load_llm():
    from langchain_openai import ChatOpenAI

    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    return ChatOpenAI(
        base_url=RAG_CONFIG["hf_base_url"],
        api_key=token,
        model=RAG_CONFIG["llm_model"],
        temperature=RAG_CONFIG["llm_temperature"],
        max_tokens=RAG_CONFIG["llm_max_tokens"],
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def predict_style(cv_bundle, image: Image.Image, top_k=3):
    import torch

    model = cv_bundle["model"]
    transform = cv_bundle["transform"]
    device = cv_bundle["device"]
    class_names = cv_bundle["class_names"]

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    elapsed_ms = (time.time() - start) * 1000

    top_idx = probs.argsort()[::-1][:top_k]
    return {
        "predicted_style": class_names[top_idx[0]],
        "confidence": float(probs[top_idx[0]]),
        "top_k": [(class_names[i], float(probs[i])) for i in top_idx],
        "inference_time_ms": round(elapsed_ms, 2),
    }


def format_context(docs):
    if not docs:
        return "(no relevant context retrieved)"
    return "\n\n".join(f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in docs)


def generate_answer(vectorstore, llm, question, chat_history, cv_context=None):
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser

    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=RAG_CONFIG["retrieval_k"]
    )
    relevant = [d for d, s in docs_and_scores if s >= RAG_CONFIG["relevance_threshold"]]

    context_text = format_context(relevant)
    if cv_context:
        context_text = f"[CV Prediction]\n{json.dumps(cv_context, indent=2)}\n\n{context_text}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context_text,
        "chat_history": chat_history,
        "input": question,
    })
    return response, relevant


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}
if "lc_history" not in st.session_state:
    from langchain_core.chat_history import InMemoryChatMessageHistory
    st.session_state.lc_history = InMemoryChatMessageHistory()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏛️ NHPT Heritage Assistant")
    st.caption("CV classification + grounded RAG chat, powered by Llama 3.1 8B via Hugging Face.")

    hf_token_present = bool(os.environ.get("HF_TOKEN"))
    if hf_token_present:
        st.success("HF_TOKEN loaded ✓")
    else:
        st.error("HF_TOKEN not found")
        st.caption("Create a `.env` file next to this app with:\n\n`HF_TOKEN=hf_your_real_token_here`")

    st.divider()
    st.subheader("1. Upload a photo")
    uploaded_file = st.file_uploader("Building / heritage site photo", type=["jpg", "jpeg", "png"])

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages = []
        from langchain_core.chat_history import InMemoryChatMessageHistory
        st.session_state.lc_history = InMemoryChatMessageHistory()
        st.session_state.last_prediction = None
        st.session_state.uploaded_image = None
        st.rerun()

    with st.expander("⚙️ Settings"):
        RAG_CONFIG["retrieval_k"] = st.slider("Chunks retrieved (k)", 1, 8, RAG_CONFIG["retrieval_k"])
        RAG_CONFIG["relevance_threshold"] = st.slider(
            "Relevance threshold", 0.0, 1.0, RAG_CONFIG["relevance_threshold"], 0.05
        )
        RAG_CONFIG["low_confidence_threshold"] = st.slider(
            "Low-confidence hedge threshold", 0.0, 1.0, RAG_CONFIG["low_confidence_threshold"], 0.05
        )


# ---------------------------------------------------------------------------
# Load resources (with friendly error handling)
# ---------------------------------------------------------------------------
try:
    vectorstore, n_docs, n_chunks = load_vectorstore()
    kb_ready = True
except Exception as e:
    kb_ready = False
    st.error(f"Failed to load the knowledge base / vector store: {e}")

cv_bundle = load_cv_model()
llm = load_llm() if hf_token_present else None

st.title("Heritage Style Assistant")

if kb_ready:
    st.caption(f"Knowledge base: {n_docs} documents → {n_chunks} chunks indexed.")
if cv_bundle is None:
    st.warning(
        "CV model files not found (`models/best_resnet50.pth`, `models/model_metadata.json`). "
        "Image classification is disabled — you can still ask text-only questions."
    )

# ---------------------------------------------------------------------------
# Image upload -> CV prediction
# ---------------------------------------------------------------------------
if uploaded_file is not None and cv_bundle is not None:
    image = Image.open(uploaded_file)
    st.session_state.uploaded_image = image

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.image(image, caption="Uploaded photo", use_container_width=True)

    with col2:
        if st.button("🔍 Analyze this photo", type="primary"):
            try:
                with st.spinner("Running CV classifier..."):
                    prediction = predict_style(cv_bundle, image)
                st.session_state.last_prediction = prediction
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        pred = st.session_state.last_prediction
        if pred:
            conf = pred["confidence"]
            st.metric("Predicted style", pred["predicted_style"], f"{conf:.0%} confidence")
            if conf < RAG_CONFIG["low_confidence_threshold"]:
                st.warning("Low confidence — the assistant will hedge and mention alternatives.")
            st.write("**Top candidates:**")
            for name, p in pred["top_k"]:
                st.progress(p, text=f"{name} — {p:.0%}")
            st.caption(f"Inference time: {pred['inference_time_ms']} ms")

st.divider()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.subheader("2. Ask about it")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

placeholder = (
    "Ask a follow-up about the analyzed photo, or ask any question about the heritage styles..."
)
user_question = st.chat_input(placeholder)

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if not kb_ready:
            reply = "The knowledge base failed to load, so I can't answer right now. Check the error above."
            st.markdown(reply)
        elif llm is None:
            reply = (
                "I can't reach the language model because `HF_TOKEN` is missing. "
                "Add it to a `.env` file next to this app and restart."
            )
            st.markdown(reply)
        else:
            with st.spinner("Thinking..."):
                try:
                    cv_context = st.session_state.last_prediction  # attach most recent prediction, if any
                    reply, sources = generate_answer(
                        vectorstore, llm, user_question,
                        st.session_state.lc_history.messages,
                        cv_context=cv_context,
                    )
                    st.markdown(reply)
                    if sources:
                        with st.expander("Sources used"):
                            for d in sources:
                                st.caption(f"📄 {d.metadata['source']}")
                except Exception as e:
                    reply = (
                        "I'm having trouble reaching the heritage assistant right now "
                        f"({type(e).__name__}). Please try again shortly."
                    )
                    st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.lc_history.add_user_message(user_question)
    st.session_state.lc_history.add_ai_message(reply)

if not st.session_state.messages:
    st.info(
        "Upload a photo and click **Analyze this photo**, then ask a question — "
        "or just start typing below for a text-only question about the six heritage styles."
    )
