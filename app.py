from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time

# Create necessary directories if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')

# Initialize session state variables
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None  # Initialize as None

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app title
st.title("PDF Chatbot")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Process the uploaded PDF file
if uploaded_file is not None:
    file_path = os.path.join("files", uploaded_file.name)
    
    # Check if the file has already been processed
    if not os.path.isfile(file_path):
        with st.status("Analyzing your document..."):
            # Save the uploaded file
            bytes_data = uploaded_file.read()
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            
            # Load the PDF file
            loader = PyPDFLoader(file_path)
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            # Create and persist the vector store
            try:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(model="mistral"),
                    persist_directory='jj'
                )
                st.session_state.vectorstore.persist()
                st.success("Vector store created successfully!")
            except Exception as e:
                st.error(f"Error creating vector store: {e}")

    # Initialize the retriever
    if st.session_state.vectorstore is not None:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    else:
        st.error("Vector store is not initialized. Please upload a valid PDF file.")

    # Initialize the QA chain if not already initialized
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        # Add user message to chat history
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                try:
                    # Get response from the QA chain
                    response = st.session_state.qa_chain(user_input)
                    full_response = response.get('result', 'No response generated.')
                except Exception as e:
                    full_response = f"An error occurred: {str(e)}"
            
            # Stream the response with a typing effect
            message_placeholder = st.empty()
            for chunk in full_response.split():
                message_placeholder.markdown(chunk + " ")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)

        # Add assistant message to chat history
        chatbot_message = {"role": "assistant", "message": full_response}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.write("Please upload a PDF file.")
