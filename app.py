import streamlit as st

from dotenv import load_dotenv

from rag import RAGSystem


load_dotenv()


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="centered"
)


# --------------------------------
# Simple UI
# --------------------------------

st.title("📄 RAG Assistant")

st.caption(
    "Upload a PDF and ask questions about it."
)


# --------------------------------
# Create RAG system
# --------------------------------

if "rag" not in st.session_state:

    st.session_state.rag = RAGSystem()


if "document_loaded" not in st.session_state:

    st.session_state.document_loaded = False


if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------
# PDF Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------
# Process PDF
# --------------------------------

if uploaded_file is not None:

    if not st.session_state.document_loaded:

        with st.spinner(
            "Reading and processing your document..."
        ):

            pages, chunks = (
                st.session_state.rag.process_pdf(
                    uploaded_file
                )
            )


        st.session_state.document_loaded = True

        st.success(
            f"Document processed successfully — "
            f"{pages} pages, {chunks} chunks."
        )


# --------------------------------
# Chat history
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# --------------------------------
# Chat input
# --------------------------------

question = st.chat_input(
    "Ask something about your document..."
)


if question:

    if not st.session_state.document_loaded:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        # User message

        with st.chat_message("user"):

            st.write(question)


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # AI response

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = st.session_state.rag.ask(
                    question
                )

            st.write(answer)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )