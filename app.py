import streamlit as st
import os

# Configure the page
st.set_page_config(page_title="Aya Burmese Expedition", page_icon="🇲🇲", layout="wide")

# Main page content
st.title("Aya Burmese Expedition 🇲🇲")
st.markdown(
    """
Welcome to the Aya Burmese Expedition dashboard! This dashboard allows you to:
1. 📸 Take snapshots of documents using your camera or upload images
2. 🔍 Extract text from Burmese documents

Use the sidebar to navigate between different pages.
"""
)

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Create directories for different file types
if not os.path.exists("data/pdf"):
    os.makedirs("data/pdf")
if not os.path.exists("data/pending"):
    os.makedirs("data/pending")
