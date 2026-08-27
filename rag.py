from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import tempfile


class RAGSystem:

    def __init__(self):

        # Embedding model
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed"
        )

        # LLM
        self.llm = ChatMistralAI(
            model="mistral-small-2506"
        )

        # Vector database will be created
        # after the user uploads a PDF
        self.vectorstore = None

    def process_pdf(self, uploaded_file):

        # --------------------------------
        # 1. Save uploaded PDF temporarily
        # --------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            pdf_path = temp_file.name


        # --------------------------------
        # 2. Load PDF
        # --------------------------------

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()


        # --------------------------------
        # 3. Split into chunks
        # --------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(
            documents
        )


        # --------------------------------
        # 4. Create vector database
        # --------------------------------

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )


        return len(documents), len(chunks)


    def ask(self, question):

        if self.vectorstore is None:

            return "Please upload a document first."


        # --------------------------------
        # 5. Retrieve relevant chunks
        # --------------------------------

        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 4
            }
        )

        relevant_docs = retriever.invoke(
            question
        )


        # --------------------------------
        # 6. Create context
        # --------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
        )


        # --------------------------------
        # 7. Prompt
        # --------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful document assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer cannot be found in the
context, say:

"I could not find the answer in the document."

Do not make up information.
"""
                ),

                (
                    "human",
                    """
Context:

{context}

Question:

{question}
"""
                )
            ]
        )


        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )


        # --------------------------------
        # 8. Generate answer
        # --------------------------------

        response = self.llm.invoke(
            final_prompt
        )


        return response.content