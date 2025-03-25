import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import json
import os
from datetime import datetime
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Any
import shutil

# Page config
st.set_page_config(page_title="Document OCR", page_icon="📸", layout="wide")

st.title("📸 Document OCR")

# Initialize session state
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "edited_text" not in st.session_state:
    st.session_state.edited_text = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Language selection
st.sidebar.header("OCR Configuration")


def process_image(image):
    """Process image for OCR"""
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    config_str = "--dpi 100"
    text = pytesseract.image_to_string(
        pil_image, lang="myan", config=config_str
    )
    return text


def save_results(image, text, metadata):
    """Save results to local directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"document_{timestamp}"

    # Save image
    image_path = os.path.join("data", f"{filename}.jpg")
    cv2.imwrite(image_path, image)

    # Save metadata
    metadata_path = os.path.join("data", f"{filename}.json")
    metadata["image_path"] = image_path
    metadata["text"] = text
    metadata["language"] = "myan"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return filename


def save_for_later(file):
    """Save file for later processing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pending_{timestamp}"

    # Create pending directory if it doesn't exist
    pending_dir = os.path.join("data", "pending")
    if not os.path.exists(pending_dir):
        os.makedirs(pending_dir)

    # Save original file
    file_path = os.path.join(pending_dir, f"{filename}_{file.name}")
    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "original_filename": file.name,
        "file_path": file_path,
        "status": "pending",
        "language": "myan",
    }

    metadata_path = os.path.join(pending_dir, f"{filename}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return filename


def process_single_file(file) -> Tuple[str, Any, Dict]:
    """Process a single file and return its results"""
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Process image
    text = process_image(image)

    # Create metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "language": "myan",
        "original_text": text,
        "edited_text": text,
        "source": "upload",
        "filename": file.name,
    }

    return text, image, metadata


# Create tabs for different input methods
tab1, tab2 = st.tabs(["📸 Camera Capture", "📁 File Upload"])

with tab1:
    st.header("Take a Photo")
    camera_image = st.camera_input("Capture document image")

    if camera_image:
        # Convert to OpenCV format
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Show preview
        st.image(image, caption="Captured Image", use_container_width=True)

        # Confirm button
        if st.button("Process Captured Image"):
            with st.spinner("Processing image..."):
                # Perform OCR
                text = process_image(image)
                st.session_state.captured_image = image
                st.session_state.ocr_text = text
                st.session_state.edited_text = text
                st.session_state.source = "camera"
                st.success("Image processed successfully!")

with tab2:
    st.header("Upload Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    # Store uploaded files in session state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

        # Display preview of uploaded files
        st.subheader("Selected Files")
        for file in uploaded_files:
            st.write(f"📄 {file.name}")

        # Create columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            # Process button
            if st.button("Process Selected Files", type="primary"):
                # Create a progress bar for overall processing
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process files in parallel
                with ThreadPoolExecutor(
                    max_workers=min(4, len(uploaded_files))
                ) as executor:
                    futures = []
                    for file in uploaded_files:
                        if file.name not in st.session_state.processing_status:
                            st.session_state.processing_status[file.name] = "pending"
                            futures.append(
                                (executor.submit(process_single_file, file), file)
                            )

                    # Update progress as files are processed
                    completed = 0
                    for future, file in futures:
                        try:
                            text, image, metadata = future.result()
                            completed += 1
                            progress = completed / len(uploaded_files)
                            progress_bar.progress(progress)
                            status_text.text(
                                f"Processed {completed} of {len(uploaded_files)} files"
                            )

                            # Store results in session state
                            st.session_state.processed_files.append(
                                {
                                    "name": metadata["filename"],
                                    "text": text,
                                    "image": image,
                                    "metadata": metadata,
                                }
                            )
                            st.session_state.processing_status[metadata["filename"]] = (
                                "completed"
                            )
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            st.session_state.processing_status[file.name] = "error"

        with col2:
            # Process Later button
            if st.button("Process Later", type="secondary"):
                with st.spinner("Saving files for later processing..."):
                    for file in uploaded_files:
                        try:
                            filename = save_for_later(file)
                            st.success(f"Saved {file.name} for later processing")
                        except Exception as e:
                            st.error(f"Error saving {file.name}: {str(e)}")

    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file_data in st.session_state.processed_files:
            with st.expander(f"📄 {file_data['name']}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        file_data["image"],
                        caption="Document Image",
                        use_container_width=True,
                    )
                with col2:
                    edited_text = st.text_area(
                        "Extracted Text",
                        file_data["text"],
                        key=f"text_{file_data['name']}",
                        height=200,
                    )
                    if st.button("Save", key=f"save_{file_data['name']}"):
                        file_data["metadata"]["edited_text"] = edited_text
                        filename = save_results(
                            file_data["image"], edited_text, file_data["metadata"]
                        )
                        st.success(f"Saved {file_data['name']} as {filename}")

# Display and edit results for camera capture
if st.session_state.captured_image is not None:
    st.header("Edit Results")

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            st.session_state.captured_image,
            caption="Document Image",
            use_container_width=True,
        )

    with col2:
        st.text_area(
            "Extracted Text",
            st.session_state.edited_text,
            height=400,
            key="text_editor",
        )
        st.session_state.edited_text = st.session_state.text_editor

    # Save button
    if st.button("Save Results"):
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "language": "myan",
            "original_text": st.session_state.ocr_text,
            "edited_text": st.session_state.edited_text,
            "source": st.session_state.get("source", "camera"),
        }

        filename = save_results(
            st.session_state.captured_image, st.session_state.edited_text, metadata
        )
        st.success(f"Results saved successfully! (Filename: {filename})")

        # Reset session state
        st.session_state.captured_image = None
        st.session_state.ocr_text = None
        st.session_state.edited_text = None
