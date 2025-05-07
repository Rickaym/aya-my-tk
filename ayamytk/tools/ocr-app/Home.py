import streamlit as st
import os

# Configure the page
st.set_page_config(page_title="Burmese OCR Dashboard", page_icon="📝", layout="wide")

# Main page content
st.title("Burmese OCR Dashboard 📝")
st.markdown(
    """
Welcome to the Burmese OCR Dashboard! This application allows you to:
1. 📸 Take snapshots of documents using your camera
2. 🔍 Process the images for Burmese text recognition
3. ✏️ Edit and verify the extracted text
4. 💾 Save the results with metadata

Use the sidebar to navigate between different pages.
"""
)

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
