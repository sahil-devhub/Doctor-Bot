from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os

# Step 1: Setup Mistral LLM with HuggingFace API
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")


def load_llm(HUGGINGFACE_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

# Step 2: Connect LLM with FAISS and Create chain
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

# Load the FAISS vector store
DB_FAISS_PATH = "vectorstores/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",  # Use the 'stuff' chain type for simplicity
    retriever=db.as_retriever(search_kwargs={"k": 3}),  # Return 3 top relevant database chunks
    return_source_documents=True,  # Tell where the answer is coming from
    chain_type_kwargs={"prompt": set_custom_prompt()}  # Uses exact prompt template
)

# Step 3: Invoke with a single query
user = input("Ask a question: ")
result = qa_chain.invoke({'query': user})
print(f"Answer: {result['result']}")
print(f"Source Documents: {result['source_documents']}")