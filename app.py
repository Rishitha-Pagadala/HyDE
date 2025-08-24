# app.py
import streamlit as st
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient

st.set_page_config(page_title="HyDE RAG", page_icon="📘", layout="centered")

HERE = Path(__file__).parent
INDEX_DIR = HERE / "faiss_index"
IDX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "index_meta.pkl"
EMBED_MODEL_NAME = "sentence-transformers/sentence-t5-large"
K = 3  # default top-k retriever


# ---- Loaders ----
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_faiss_index():
    if not IDX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("faiss_index not found. Run build_index.py locally and commit 'faiss_index/' to the repo.")
    index = faiss.read_index(str(IDX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

@st.cache_resource(show_spinner=False)
def get_hf_client():
    if "HF_TOKEN" not in st.secrets:
        st.error("HF token not found. Add HF_TOKEN to Streamlit secrets.")
        st.stop()
    return InferenceClient(model="google/gemma-2-9b", token=st.secrets["HF_TOKEN"])


# ---- Utils ----
def embed_query(query: str, embedder: SentenceTransformer):
    vec = embedder.encode([query], convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # normalize
    return vec.astype("float32")

def search(index, meta, qvec, top_k=K):
    D, I = index.search(qvec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append(meta[idx]["text"])
    return results

def make_prompt(context_chunks, question):
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the user's question STRICTLY using only the CONTEXT below. 
If the answer is not contained in the context, reply exactly: "The context does not provide this information."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""


# ---- UI ----
st.title("HyDE RAG")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about the PDF content", "")

if st.button("Ask") and query.strip():
    with st.spinner("Searching and generating answer..."):
        embedder = load_embedder()
        index, metadata = load_faiss_index()
        client = get_hf_client()

        qvec = embed_query(query, embedder)
        context_chunks = search(index, metadata, qvec, top_k=K)
        prompt = make_prompt(context_chunks, query)

        try:
            resp = client.text_generation(prompt, max_new_tokens=256, do_sample=False, temperature=0.2)
        except Exception as e:
            st.error(f"Error from HF Inference API: {e}")
            raise

        # Parse response
        answer = ""
        if isinstance(resp, list) and len(resp) > 0:
            first = resp[0]
            if isinstance(first, dict):
                answer = first.get("generated_text") or first.get("text") or str(first)
            else:
                answer = str(first)
        elif isinstance(resp, dict):
            answer = resp.get("generated_text") or resp.get("text") or str(resp)
        else:
            answer = str(resp)

        # Save to history
        st.session_state.history.append({"question": query, "answer": answer})


# ---- Show conversation history ----
if st.session_state.history:
    st.markdown("### 💬 Conversation History")
    for i, qa in enumerate(st.session_state.history, start=1):
        st.markdown(f"**Q{i}:** {qa['question']}")
        st.markdown(f"**A{i}:** {qa['answer']}")
        st.markdown("---")
