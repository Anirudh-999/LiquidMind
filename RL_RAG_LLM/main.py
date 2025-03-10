import json
import os
import numpy as np
import streamlit as st
import faiss
import PyPDF2
import pdfplumber  # Better for complex layouts
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import ollama
from pdf2image import convert_from_path  # For extracting images from PDF
from PIL import Image  # For handling images
import pytesseract  # For OCR (text extraction from images)

# Load environment variables
load_dotenv()

# Initialize Azure Chat OpenAI
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0.7
)

embedding_model = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")

def extract_text_and_images_from_pdf(pdf_file):
    """Extract text and images from uploaded PDF file."""
    text = ""
    images = []

    # Try extracting text using PyPDF2
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")

    # If no text is extracted, try pdfplumber for complex layouts
    if not text:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")

    # Extract images using pdf2image
    try:
        images = convert_from_path(pdf_file.name)  # Convert PDF pages to images
    except Exception as e:
        print(f"Error extracting images: {e}")
    
    return text, images

def extract_text_from_image(image):
    """Extract text from an image using OCR (Tesseract)."""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def chunk_text(text: str, max_tokens: int = 2000, overlap: int = 200) -> List[str]:
    """Efficiently split text into chunks with overlap."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    
    num_tokens = len(tokens)
    return [encoding.decode(tokens[i:min(i + max_tokens, num_tokens)]) 
            for i in range(0, num_tokens, max_tokens - overlap)]

def get_embedding(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple text chunks at once using HuggingFaceEmbeddings."""
    try:
        embeddings = embedding_model.embed_documents(texts)  # Use HuggingFaceEmbeddings
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return [None] * len(texts)

def create_faiss_index(text_chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = get_embedding(text_chunks)  # Get embeddings for the text chunks
    embedding_array = np.array(embeddings, dtype=np.float32)
    faiss_index = FAISS.from_texts(
        texts=text_chunks, 
        embedding=embedding_model
    )
    return faiss_index

def query_llm(query, context, feedback_history=None):
    """Query the DeepSeek 1.5B model hosted on Ollama."""
    feedback_prompt = ""
    if feedback_history:
        feedback_prompt = "\n\nPrevious Feedback:\n"
        for feedback in feedback_history:
            feedback_prompt += f"- Question: {feedback['query']}\n  Feedback: {feedback['comments']}\n"

    prompt = f"""Context: {context}

    Question: {query}{feedback_prompt}

    Instructions:
    - Provide a clear, concise answer based on the context
    - Focus only on relevant information
    - If the answer isn't in the context, say "I cannot answer this based on the provided context"
    - Keep the response under 3-4 sentences unless absolutely necessary
    - Use previous feedback to improve the answer

    Answer:"""
    response = ollama.chat(
        model="deepseek-r1:1.5b", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

def load_feedback(filepath="feedback_log.json") -> List[Dict]:
    """Load feedback from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_feedback(feedback_data, filepath="feedback_log.json"):
    """Save feedback to a JSON file."""
    data = load_feedback(filepath)
    data.append(feedback_data)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# Initialize session state for feedback storage if not already present
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = load_feedback()

st.title("📄 RAG-powered PDF Q&A with DeepSeek 1.5B (with Feedback Loop)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        # Extract text and images from the PDF
        text, images = extract_text_and_images_from_pdf(uploaded_file)
        
        # Extract text from images using OCR
        image_texts = [extract_text_from_image(image) for image in images]
        combined_text = text + "\n".join(image_texts)  # Combine text and OCR-extracted text
        
        if not combined_text:
            st.error("No text or images could be extracted from the PDF. Please ensure the PDF is not scanned or encrypted.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text_chunks = text_splitter.split_text(combined_text)
            
            # Create FAISS index with embeddings
            faiss_index = create_faiss_index(text_chunks)
            st.success("PDF processed! You can now ask questions.")

    def ask_question():
        """Function to handle the question asking process."""
        while True:  # Loop to continue asking questions
            query = st.text_input("Ask a question about the document:")
            if not query:  # If no query is provided, break the loop
                break

            with st.spinner("Searching for answers..."):
                docs = faiss_index.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                answer = query_llm(query, context, st.session_state.feedback_history)
                st.markdown(f"### Answer:\n{answer}")

            # --- Feedback Loop ---
            st.markdown("### Provide Feedback")
            feedback_comments = st.text_area("Your feedback (optional):", "")

            if st.button("Submit Feedback"):
                feedback_data = {
                    "query": query,
                    "context": context,
                    "answer": answer,
                    "comments": feedback_comments
                }
                st.session_state.feedback_history.append(feedback_data)
                save_feedback(feedback_data)
                st.success("Thank you for your feedback!")

                # Clear the feedback input field after submission
                feedback_comments = ""  # Clear comments

            # Allow the user to ask more questions
            if not st.button("Ask another question"):
                break  # Break the loop if the user does not want to ask another question

    # Start the question asking process
    ask_question()
