import streamlit as st  # For building the Streamlit app
from langchain_huggingface import HuggingFaceEmbeddings  # For embeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # For LLM and chat model
from langchain.chains import RetrievalQA  # For question answering chain
from langchain_community.vectorstores import FAISS  # For vector store
from langchain.prompts import ChatPromptTemplate  # For custom prompts
import os


@st.cache_resource  # Cache the vector store to avoid reloading
def load_vector_store():
    DB_FAISS_PATH = "vectorstores/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True) # allow_dangerous_deserialization=True, defines you trust this data-source, without using it you will get valueerror or crash code.
    return db


def load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def set_custom_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            """Use the pieces of information provided in the context to answer user's questions.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Don't provide anything out of the given context.

            Context: {context}"""),
            ("human", "Question: {question}"),
        ]
    )
    return prompt


def main():
    st.title("Doctor Bot")   # Set the title of the Streamlit app

    if "messages" not in st.session_state:   # Check if messages are in session state
        st.session_state.messages = [] # Initialize session state for messages

    for message in st.session_state.messages:  # Display previous messages
        if message['role'] == 'user':    # User message
            st.chat_message("user").markdown([message['content']])    # Display user message in the chat
        else:   # Bot message
            st.chat_message("assistant").markdown([message['content']])   # Display bot message in the chat

    input_text = st.chat_input("Enter your question:")  # Input field for user question

    if input_text:
        st.chat_message("user").markdown(input_text) # Display user input in the chat
        st.session_state.messages.append({"role": "user", "content": input_text})

        # Load environment variables and vector store
        HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        HF_TOKEN = os.getenv("HF_TOKEN")
        db = load_vector_store()
        llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

        try:
            # Create the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3}),  # Return 3 top relevant database chunks
                return_source_documents=True,  # Tell where the answer is coming from
                chain_type_kwargs={"prompt": set_custom_prompt()}  # Use exact prompt template
            )

            # Get response from the QA chain
            result = qa_chain.invoke({'query': input_text})
            response = f"Doctor Bot: {result['result']}"
            st.chat_message("assistant").markdown(response)   # Display the response in the chat
            st.session_state.messages.append({"role": "assistant", "content": response})  # Store the response in session state

            # Optionally display source documents
            st.write("Source Documents:")
            for doc in result['source_documents']:
                page_data = doc.metadata.get("page")
                source_String = f"Page{page_data}: " if page_data is not None else "Source Document: "
                st.write(f"{source_String}{doc.page_content}")

        except Exception as e:
            response = f"Doctor Bot: An error occurred: {str(e)}"
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()