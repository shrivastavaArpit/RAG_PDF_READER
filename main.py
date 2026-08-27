from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# --------------------------------
# 1. Embedding model
# --------------------------------

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)


# --------------------------------
# 2. Load vector database
# --------------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# --------------------------------
# 3. Create retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# --------------------------------
# 4. LLM
# --------------------------------

llm = ChatMistralAI(
    model="mistral-small-2506"
)


# --------------------------------
# 5. Prompt
# --------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say:

"I could not find the answer in the document."
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


# --------------------------------
# 6. RAG loop
# --------------------------------

print("\nRAG System Ready!")
print("Type 0 to exit.\n")


while True:

    query = input("You: ")

    if query == "0":
        break


    # Retrieve relevant documents
    docs = retriever.invoke(query)


    # Combine retrieved documents
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # Create final prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # Generate answer
    response = llm.invoke(final_prompt)


    print("\nAI:", response.content)
    print()