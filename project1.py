# import os
# from dotenv import load_dotenv
# from langchain import PromptTemplate, LLMChain
# from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import TextLoader
# from langchain.embeddings import HuggingFaceEmbeddings


# #  Load API keys

# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# def embed_docs():
#     folder_path = "data/medical_docs"
#     os.makedirs(folder_path, exist_ok=True)

#     # Create a default doc if none exist
#     if not any(fname.endswith(".txt") for fname in os.listdir(folder_path)):
#         with open(os.path.join(folder_path, "first_aid.txt"), "w") as f:
#             f.write(
#                 "For mild fever, rest and hydration are important. "
#                 "Over-the-counter fever reducers may help. "
#                 "Seek medical attention if fever exceeds 102°F or lasts more than 3 days. "
#                 "For sore throats, warm salt water gargles and hydration can help."
#             )

#     print("📘 Embedding documents...")
#     embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     docs = []

#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             loader = TextLoader(os.path.join(folder_path, filename))
#             docs.extend(loader.load())

#     db = Chroma.from_documents(docs, embedding_fn, persist_directory=folder_path)
#     db.persist()
#     print("✅ Documents embedded successfully.")
#     return folder_path


# # ───────────────────────────────────────────────
# #  Create Groq LLM
# # ───────────────────────────────────────────────
# def get_llm(model="mixtral-8x7b-32768", temperature=0.3):
#     return ChatGroq(
#         temperature=temperature,
#         model=model,
#         groq_api_key=GROQ_API_KEY,
#     )


# # ───────────────────────────────────────────────
# #  Symptom Analysis Chain
# # ───────────────────────────────────────────────
# def create_symptom_chain():
#     prompt = PromptTemplate(
#         input_variables=["symptoms", "history"],
#         template="""
# You are a cautious and helpful home healthcare assistant.

# Patient reports: {symptoms}
# Medical history: {history}

# Provide:
# 1. Likely causes or factors (without making a medical diagnosis)
# 2. Home-care or lifestyle advice
# 3. When they should consult a doctor
# 4. Keep tone friendly and supportive.
# """
#     )

#     llm = get_llm()
#     return LLMChain(prompt=prompt, llm=llm)


# # ───────────────────────────────────────────────
# #  Retrieval Chain
# # ───────────────────────────────────────────────
# def create_retrieval_chain():
#     folder_path = embed_docs()
#     llm = get_llm(temperature=0.2)
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     db = Chroma(
#         persist_directory=folder_path,
#         embedding_function=embeddings
#     )
#     retriever = db.as_retriever(search_kwargs={"k": 2})
#     return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# # ───────────────────────────────────────────────
# #  Main Health Assistant Loop
# # ───────────────────────────────────────────────
# def main():
#     print("🩺 AI Health Assistant (Groq Edition)\nType 'exit' to quit.\n")

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     symptom_chain = create_symptom_chain()
#     retrieval_chain = create_retrieval_chain()

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break

#         past = memory.load_memory_variables({}).get("chat_history", "")

#         # Step 1 — Analyze symptoms
#         symptom_analysis = symptom_chain.run({
#             "symptoms": user_input,
#             "history": past
#         })

#         # Step 2 — Retrieve knowledge if informational
#         if any(word in user_input.lower() for word in ["what", "how", "why", "can", "should"]):
#             retrieval_response = retrieval_chain.run(user_input)
#             final_response = f"{symptom_analysis}\n\n📚 Relevant info:\n{retrieval_response}"
#         else:
#             final_response = symptom_analysis

#         print(f"\nAI: {final_response}\n")
#         memory.save_context({"user": user_input}, {"ai": final_response})


# # ───────────────────────────────────────────────
# #  Run
# # ───────────────────────────────────────────────
# if __name__ == "__main__":
#     main()


# streamlit_app.py
import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------
# Utilities / caching
# -----------------------
@st.cache_resource(show_spinner=False)
def get_embedding_fn():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm(model="mixtral-8x7b-32768", temperature=0.3):
    if not GROQ_API_KEY:
        # We'll raise at call-time in UI rather than here, but return a placeholder
        return None
    return ChatGroq(
        temperature=temperature,
        model=model,
        groq_api_key=GROQ_API_KEY,
    )

@st.cache_data(show_spinner=False)
def embed_docs(folder_path: str):
    """
    Read .txt files from folder_path and create/persist a Chroma DB.
    If none exist, create a default first_aid.txt (same behavior as your CLI).
    Returns the folder_path used.
    """
    os.makedirs(folder_path, exist_ok=True)

    # Create a default doc if none exist
    if not any(fname.endswith(".txt") for fname in os.listdir(folder_path)):
        with open(os.path.join(folder_path, "first_aid.txt"), "w", encoding="utf-8") as f:
            f.write(
                "For mild fever, rest and hydration are important. "
                "Over-the-counter fever reducers may help. "
                "Seek medical attention if fever exceeds 102°F or lasts more than 3 days. "
                "For sore throats, warm salt water gargles and hydration can help."
            )

    embedding_fn = get_embedding_fn()
    docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename), encoding="utf-8")
            docs.extend(loader.load())

    db = Chroma.from_documents(docs, embedding_fn, persist_directory=folder_path)
    db.persist()
    return folder_path

@st.cache_resource(show_spinner=False)
def create_retrieval_chain_cached(folder_path: str):
    """
    Build a RetrievalQA chain and cache it for the given folder path.
    """
    llm = get_llm(temperature=0.2)
    if llm is None:
        return None

    embeddings = get_embedding_fn()
    db = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# -----------------------
# Chains (symptom analysis)
# -----------------------
def create_symptom_chain():
    prompt = PromptTemplate(
        input_variables=["symptoms", "history"],
        template="""
You are a cautious and helpful home healthcare assistant.

Patient reports: {symptoms}
Medical history: {history}

Provide:
1. Likely causes or factors (without making a medical diagnosis)
2. Home-care or lifestyle advice
3. When they should consult a doctor
4. Keep tone friendly and supportive.
"""
    )

    llm = get_llm()
    return LLMChain(prompt=prompt, llm=llm)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Health Assistant (Groq + LangChain)", layout="centered")
st.title("🩺 AI Health Assistant — Groq Edition")
st.caption("A simple home-health assistant built with LangChain, Groq, and Chroma. Not a substitute for professional advice.")

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Docs folder (persist_directory)", value="data/medical_docs")
    model_name = st.text_input("Groq model", value="mixtral-8x7b-32768")
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    show_retrieval = st.checkbox("Enable retrieval from docs", value=True)
    uploaded_files = st.file_uploader("Upload .txt medical docs (optional)", accept_multiple_files=True, type=["txt"])

    st.markdown("---")
    st.markdown("**Environment**")
    st.write(f"GROQ_API_KEY set: {'Yes' if GROQ_API_KEY else 'No'}")
    st.markdown("Run: `streamlit run streamlit_app.py`")

# Handle uploaded files (save into data_dir)
if uploaded_files:
    os.makedirs(data_dir, exist_ok=True)
    for f in uploaded_files:
        target_path = os.path.join(data_dir, f.name)
        # write uploaded file to disk
        with open(target_path, "wb") as out:
            out.write(f.getbuffer())
    st.success(f"Saved {len(uploaded_files)} file(s) to `{data_dir}`. Rebuilding embeddings...")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, ai) tuples

if "symptom_chain" not in st.session_state or st.session_state.get("symptom_model") != model_name or st.session_state.get("symptom_temp") != temperature:
    # Recreate symptom chain based on current model/temperature
    # Note: get_llm uses cached ChatGroq by model/temperature only if parameters match; to keep simple we will not aggressively cache LLM variants.
    # We'll store model/temp so we rebuild when changed.
    st.session_state.symptom_model = model_name
    st.session_state.symptom_temp = temperature
    # Create a new llm factory with desired settings by temporarily overriding get_llm - simpler approach:
    st.session_state.symptom_chain = create_symptom_chain()

# Prepare retrieval chain (if enabled)
retrieval_chain = None
if show_retrieval:
    with st.spinner("Preparing document embeddings & retrieval..."):
        try:
            folder_path = embed_docs(data_dir)
            retrieval_chain = create_retrieval_chain_cached(folder_path)
            if retrieval_chain is None:
                st.error("Missing GROQ_API_KEY or LLM initialization failed for retrieval chain.")
        except Exception as e:
            st.error(f"Failed to build/retrieve embeddings: {e}")

# Chat input form
with st.form("chat_form", clear_on_submit=False):
    st.subheader("Describe symptoms")
    user_input = st.text_area("What symptoms or question do you have?", height=140)
    history_text = st.text_area("Medical history / previous notes (optional)", height=100, value="\n".join(
        f"User: {u}\nAI: {a}" for u, a in st.session_state.history[-5:]
    ))
    include_retrieval = st.checkbox("Run document retrieval (if available)", value=show_retrieval)
    submit = st.form_submit_button("Submit")

if submit:
    if not user_input.strip():
        st.warning("Please enter some symptoms or a question before submitting.")
    else:
        # sanity: ensure API key present
        if GROQ_API_KEY is None or GROQ_API_KEY == "":
            st.error("GROQ_API_KEY is not set. Set it in your environment or .env before running this app.")
        else:
            # Build LLM with current settings (re-create ChatGroq with temperature/model)
            llm = ChatGroq(temperature=temperature, model=model_name, groq_api_key=GROQ_API_KEY)
            symptom_prompt = PromptTemplate(
                input_variables=["symptoms", "history"],
                template="""
You are a cautious and helpful home healthcare assistant.

Patient reports: {symptoms}
Medical history: {history}

Provide:
1. Likely causes or factors (without making a medical diagnosis)
2. Home-care or lifestyle advice
3. When they should consult a doctor
4. Keep tone friendly and supportive.
"""
            )
            symptom_chain = LLMChain(prompt=symptom_prompt, llm=llm)

            # Run symptom analysis
            with st.spinner("Analyzing symptoms..."):
                try:
                    # For history, prefer the user-provided history_text; fall back to session history
                    history_for_llm = history_text if history_text.strip() else "\n".join(f"User: {u}\nAI: {a}" for u, a in st.session_state.history[-5:])
                    symptom_analysis = symptom_chain.run({
                        "symptoms": user_input,
                        "history": history_for_llm
                    })
                except Exception as e:
                    st.error(f"Symptom analysis failed: {e}")
                    symptom_analysis = "Error: symptom analysis failed."

            retrieval_response = None
            if include_retrieval and retrieval_chain is not None:
                with st.spinner("Searching documents..."):
                    try:
                        # retrieval_chain expects a plain query string
                        retrieval_response = retrieval_chain.run(user_input)
                    except Exception as e:
                        st.error(f"Retrieval failed: {e}")
                        retrieval_response = None

            # Build final response
            final_response = symptom_analysis
            if retrieval_response:
                final_response = f"{symptom_analysis}\n\n📚 Relevant info:\n{retrieval_response}"

            # Save to session history
            st.session_state.history.append((user_input, final_response))

            # Display
            st.markdown("### 📝 Assistant response")
            st.write(final_response)

# Show chat history
if st.session_state.history:
    st.markdown("---")
    st.subheader("Conversation history")
    for i, (u, a) in enumerate(reversed(st.session_state.history[-20:]), 1):
        st.markdown(f"**You:** {u}")
        st.markdown(f"**AI:** {a}")
        st.markdown("---")

# Controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear history"):
        st.session_state.history = []
        st.experimental_rerun()
with col2:
    if st.button("Rebuild embeddings (force)"):
        # Clear cached data then rebuild
        try:
            embed_docs(data_dir)
            # clear retrieval cache
            create_retrieval_chain_cached.clear()
            st.success("Rebuilt embeddings and cleared retrieval chain cache.")
        except Exception as e:
            st.error(f"Failed to rebuild embeddings: {e}")

st.markdown(
    """
---
**Notes & Next steps**

- This app is for informational, non-diagnostic purposes only. If symptoms are severe, seek medical attention.
- To run: `pip install -r requirements.txt` and then `streamlit run streamlit_app.py`.
- If you want uploads to persist long-term, use a persistent `data/medical_docs` folder (or change `Docs folder` in the sidebar).
"""
)
