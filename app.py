# streamlit_app.py (or whatever you name your main Streamlit file)

import streamlit as st
import ollama
import io
import tempfile
import os
import base64
import PyPDF2  # Ensure this is in your requirements.txt

def get_ollama_response(prompt, model="mistral"):
    """Gets a response from the Ollama model."""
    try:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True)
        full_response = ""
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                yield chunk['message']['content']
        return full_response

    except Exception as e:
        yield f"Error: {e}"
        return f"Error: {e}"

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    return text

def extract_info_from_invoice(invoice_content, user_prompt):
    """Extracts information from the invoice using Ollama and the user's prompt."""
    combined_prompt = f"Here is the invoice content:\n\n{invoice_content}\n\nUser prompt: {user_prompt}\n\nExtract and provide the requested information."

    return get_ollama_response(combined_prompt)

def display_pdf(file_bytes):
    """Displays a PDF in the Streamlit app."""
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.title("Invoice Information Extractor")

    uploaded_file = st.file_uploader("Upload an invoice (PDF)", type=["pdf"])
    user_prompt = st.text_area("Enter your prompt (e.g., 'What is the total amount?', 'What is the invoice number?', 'List all items purchased.')")

    if uploaded_file is not None and user_prompt:
        try:
            file_bytes = uploaded_file.read()

            # Display the uploaded PDF
            display_pdf(file_bytes)

            # Extract text from PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name

                invoice_text = extract_text_from_pdf(tmp_file_path)

            st.subheader("Extracted Information:")
            for chunk in extract_info_from_invoice(invoice_text, user_prompt):
                st.write(chunk)

            os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
