import streamlit as st
import json
from PIL import Image
from app import ocrcore  # Importing the OCR functions from app.py
from records import oppo
from records import edit

st.header("Invoice Scanner")
st.divider()

# Initialize session state so that when there is nothing before it will create a variable

if "page" not in st.session_state:
    st.session_state["page"] = "home"

if "invoice" not in st.session_state:
    st.session_state["invoice"] = {}

# File uploader

extracted_text = "None"

if st.session_state["page"] == "home":

    if st.button("Add new document"):
        st.session_state["page"] = "new"

    elif st.button("View past records"):
        st.session_state["page"] = "past"
        pass

    elif st.button("chat"):
        st.session_state["page"] = "chat"

    else:pass

elif st.session_state["page"] == "new":

    date = st.text_input("Enter the date in format dd/mm/yy")

    uploaded_file = st.file_uploader("Upload your invoice", type=["jpg", "png"])

    if uploaded_file is not None:
        st.write("File uploaded successfully")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        #Extract text using OCR 
        extracted_text = ocrcore(uploaded_file)
        st.write(extracted_text)
        dictionary = edit(extracted_text)
        st.write(dictionary)
        st.session_state["invoice"][date] = dictionary
        st.write(st.session_state["invoice"])
        cleaned_data = st.session_state["invoice"][date].strip("```json").strip("```").strip("'''\"").strip()
        st.write(cleaned_data)

    if st.button("Go back"):
            st.session_state["page"] = "home"    
    
elif st.session_state['page'] == "past":
    date = st.text_input("Enter the date you want to access (dd/mm/yy)")
    if date in st.session_state["invoice"]:
        invoice_data = st.session_state["invoice"][date]
        st.write(f"Invoice for {date}:")
        st.write(invoice_data)
    else:
        st.write(f"No invoice data found for {date}.")
    
    if st.button("Go back"):
            st.session_state["page"] = "home"   


elif st.session_state['page'] == "chat":
    st.write("Enter the date for details")
    date = st.text_input("Date: (format dd/mm/yy)")
    if date in st.session_state["invoice"]:
        cleaned_data = str(st.session_state["invoice"][date])
    elif date not in st.session_state["invoice"]:
        st.write(f"Date {date} not found in session_state.")
    operation = st.text_input("operation you want to perform:")
    if st.button("Go"):
        st.write(oppo(cleaned_data,operation))
        if st.button("try again"):
            st.session_state["page"] = "chat" 
    if st.button("Go back"):
        st.session_state["page"] = "home"  
    

