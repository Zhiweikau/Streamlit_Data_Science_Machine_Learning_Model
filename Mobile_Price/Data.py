import streamlit as st
import pandas as pd

# Ensure session state is initialized
if "data" not in st.session_state:
    st.session_state.data = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0 

data = st.session_state.data

# Title and Image
image_path = r"Phone_image.png"

col1, col2 = st.columns([0.2, 0.8])

with col1:
    st.image(image_path, width=80)

with col2:
    st.title("Mobile Price Classification")

st.markdown("---")
st.write("Training Dataset Uploaded")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=f"uploaded_file_{st.session_state.uploader_key}")

if uploaded_file is not None:
    # Always update session state with the new file
    data = pd.read_csv(uploaded_file)
    st.session_state.data = data  # Store updated data in session state

# Ensure table always appears when navigating back
if data is not None:
    st.write("Top 5 rows of the dataset (Persisted Data):")
    st.dataframe(data.head())

# Clear File Button
if st.button("Clear Uploaded File"):
    st.session_state.data = None  # Reset the stored data
    st.session_state.uploaded_key += 1
    st.write("Uploaded file has been cleared.")
    st.rerun()
