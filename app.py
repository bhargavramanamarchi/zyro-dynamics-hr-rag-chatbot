import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🤖")

st.title("🤖 Zyro Dynamics HR Help Desk")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def build_rag():

    loader = PyPDFDirectoryLoader("zyro-dynamics-hr-corpus")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":3}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return retriever, llm

retriever, llm = build_rag()

RAG_PROMPT = ChatPromptTemplate.from_template(
'''
You are an HR assistant for Zyro Dynamics.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
'''
)

def ask_bot(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, docs

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask an HR policy question..."
)

if question:

    st.session_state.messages.append(
        {"role":"user","content":question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    answer, docs = ask_bot(question)

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.markdown("### Sources")

        for i, doc in enumerate(docs,1):
            source = doc.metadata.get(
                "source",
                "Unknown Source"
            )

            st.markdown(
                f"{i}. {source}"
            )

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )