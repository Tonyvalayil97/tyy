import streamlit as st
import requests
import fitz  # PyMuPDF for PDF text extraction
from PIL import Image
import pytesseract

# Ollama Mistral API endpoint (replace with your setup details)
OLLAMA_URL = "http://localhost:11434/v1/ask"

def extract_text_from_pdf(file):
    # Open the PDF file and extract text from it
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_image(file):
    # Use Tesseract to extract text from image
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

def ask_mistral_model(prompt, extracted_text):
    # Query the Mistral model with the prompt and extracted invoice text
    payload = {
        "input": f"Extract details from the following invoice based on the prompt: {prompt}\nInvoice Text:\n{extracted_text}"
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json().get('text')
    else:
        return "Error in processing the request"

def main():
    st.title("Invoice Information Extractor")

    # Upload an invoice file (PDF or Image)
    uploaded_file = st.file_uploader("Upload your invoice", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Extract text based on the file type
        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            extracted_text = extract_text_from_image(uploaded_file)

        st.subheader("Extracted Text")
        st.text_area("Invoice Text", extracted_text, height=300)

        # Input prompt for the Mistral model
        prompt = st.text_input("Enter your prompt to extract info", "Please extract invoice number, amount, and weight.")

        if prompt:
            # Ask Mistral to process the extracted text
            result = ask_mistral_model(prompt, extracted_text)
            st.subheader("Extracted Information")
            st.write(result)

if __name__ == "__main__":
    main()
