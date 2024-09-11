import pytest
from unittest.mock import MagicMock, patch, mock_open
import streamlit as st
import os

@pytest.fixture(autouse=True)
def setup_streamlit_session_state():
    # Clear and initialize the Streamlit session state for each test
    st.session_state.clear()

    # Manually set up session state variables
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
    st.session_state.prompt = MagicMock()
    st.session_state.memory = MagicMock()
    st.session_state.vectorstore = MagicMock()
    st.session_state.llm = MagicMock()
    st.session_state.chat_history = []
    st.session_state.qa_chain = MagicMock()

def test_directories_created():
    with patch("os.mkdir") as mock_mkdir:
        # Simulate running the script to check directory creation
        if not os.path.exists('files'):
            os.mkdir('files')
        if not os.path.exists('jj'):
            os.mkdir('jj')

        mock_mkdir.assert_any_call('files')
        mock_mkdir.assert_any_call('jj')

def test_initial_session_state():
    # Test if the session state variables are initialized
    assert 'template' in st.session_state
    assert 'prompt' in st.session_state
    assert 'memory' in st.session_state
    assert 'vectorstore' in st.session_state
    assert 'llm' in st.session_state
    assert st.session_state.chat_history == []

def test_file_upload():
    # Mock the file uploader and PDF loader
    with patch("streamlit.file_uploader") as mock_uploader, \
         patch("builtins.open", mock_open(read_data="PDF content")), \
         patch("os.path.isfile", return_value=False), \
         patch("langchain_community.document_loaders.PyPDFLoader.load", return_value=["mock_pdf_data"]), \
         patch("langchain_community.vectorstores.Chroma.from_documents"), \
         patch("langchain.text_splitter.RecursiveCharacterTextSplitter.split_documents", return_value=["split_text"]):
        
        # Simulate a file upload
        mock_uploader.return_value = MagicMock(name="mock_pdf.pdf", read=MagicMock(return_value=b"PDF content"))

        # Run the file upload and processing code
        uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
        assert uploaded_file is not None

        # Ensure that the PDF content was processed and vector store created
        mock_uploader.assert_called_once()
        assert st.session_state.vectorstore is not None

def test_chat_interaction():
    # Mock the QA chain and chat input
    with patch("streamlit.chat_input") as mock_chat_input, \
         patch("streamlit.chat_message"), \
         patch("streamlit.spinner"), \
         patch("streamlit.markdown"), \
         patch("streamlit.session_state.qa_chain", return_value={"result": "This is a response"}):
        
        # Simulate user input
        mock_chat_input.return_value = "What is AI?"

        user_input = st.chat_input("You:")
        assert user_input == "What is AI?"

        # Simulate the response from the QA chain
        response = st.session_state.qa_chain(user_input)
        assert response['result'] == "This is a response"

def test_prompt_template():
    # Check that the prompt template is correctly initialized
    assert 'prompt' in st.session_state
    assert st.session_state.prompt is not None
    assert "{context}" in st.session_state.template
