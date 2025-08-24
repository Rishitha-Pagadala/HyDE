# app.py
import streamlit as st
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
import time

st.set_page_config(page_title="RAG PDF (Streamlit)", page_icon="🧠", layout="centered")

HERE = Path(__file__).parent
INDEX_DIR = HERE / "faiss_index"
IDX_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "index_meta.pkl"
EMBED_MODEL_NAME = "sentence-transformers/sentence-t5-large"
K = 3
MIN_SCORE = 0.25  # cosine similarity threshold (tune as needed)


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
    token = None
    # prefer streamlit secrets
    if "HF_TOKEN" in st.secrets:
        token = st.secrets["HF_TOKEN"]
    else:
        token = st.session_state.get("HF_TOKEN", None)
    if not token:
        st.error("HF token not found. Add HF_TOKEN to Streamlit secrets.")
        st.stop()
    return InferenceClient(model="google/gemma-2-9b", token=token)

def embed_query(query: str, embedder: SentenceTransformer):
    vec = embedder.encode([query], convert_to_numpy=True)
    # normalize for cosine
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vec.astype("float32")

def search(index, meta, qvec, top_k=K):
    D, I = index.search(qvec, top_k)
    # D are dot-products (since IndexFlatIP on normalized vectors)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append((meta[idx]["text"], meta[idx]["source"], float(score)))
    return results

def make_prompt(context_chunks, question):
    context_text = "\n\n---\n\n".join([c for c in context_chunks])
    prompt = f"""You are a helpful assistant. Answer the user's question STRICTLY using only the CONTEXT below. 
If the answer is not contained in the context, reply exactly: "The context does not provide this information."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
    return prompt

st.title("RAG PDF QA (Streamlit)")

st.markdown("Upload/Use the pre-built FAISS index (from the provided PDF). Answers are generated via Hugging Face Inference API. Make sure to add your HF token to Streamlit secrets as `HF_TOKEN`.")

# small UI for token override if user sets env differently (optional)
with st.expander("HF token (optional)"):
    token_input = st.text_input("Paste HF token (optional, will override secrets for this session)", type="password")
    if token_input:
        st.session_state["HF_TOKEN"] = token_input

query = st.text_input("Ask a question about the PDF content", "")
k = st.number_input("Retriever k", value=3, min_value=1, max_value=10, step=1)
threshold = st.slider("Minimum similarity (0-1)", 0.0, 1.0, float(MIN_SCORE), 0.01)

if st.button("Ask") and query.strip():
    with st.spinner("Loading resources..."):
        embedder = load_embedder()
        index, metadata = load_faiss_index()
        client = get_hf_client()

    qvec = embed_query(query, embedder)
    results = search(index, metadata, qvec, top_k=k)

    # filter by threshold
    filtered = [r for r in results if r[2] >= threshold]
    if not filtered:
        st.warning("No sufficiently similar context found. Try lowering the similarity threshold or rephrase the question.")
    else:
        # show retrieved sources
        st.markdown("**Retrieved sources (score):**")
        for i, (txt, src, sc) in enumerate(filtered, start=1):
            st.write(f"{i}. {src} — score: {sc:.4f}")

        context_chunks = [txt for txt, _, _ in filtered]
        prompt = make_prompt(context_chunks, query)

        st.markdown("### Answer")
        with st.spinner("Generating answer from model..."):
            # call HF text-generation endpoint
            # model choice: you can replace model_id with a preferred hosted model
            # using the Inference Client default model if not specified.
            try:
                resp = client.text_generation(prompt, max_new_tokens=256, do_sample=False, temperature=0.2)
            except Exception as e:
                st.error(f"Error from HF Inference API: {e}")
                raise

            # parse response (various hf client versions differ)
            answer = ""
            if isinstance(resp, list) and len(resp) > 0:
                # some clients return list of dicts
                first = resp[0]
                if isinstance(first, dict):
                    # common key
                    answer = first.get("generated_text") or first.get("text") or str(first)
                else:
                    answer = str(first)
            elif isinstance(resp, dict):
                answer = resp.get("generated_text") or resp.get("text") or str(resp)
            else:
                answer = str(resp)

        st.write(answer)

        st.markdown("#### Context snippets used (for transparency)")
        for i, c in enumerate(context_chunks, start=1):
            st.markdown(f"**Snippet {i}:**")
            st.write(c[:1000] + ("..." if len(c) > 1000 else ""))
            st.caption(filtered[i-1][1])
