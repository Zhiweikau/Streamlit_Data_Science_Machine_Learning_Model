import streamlit as st
import pandas as pd

# Ensure session state is initialized
if "data" not in st.session_state:
    st.session_state.data = None

# Title and Image
image_path = "https://github.com/Zhiweikau/Streamlit_Data_Science_Machine_Learning_Model/blob/main/Phone_image.png"

col1, col2 = st.columns([0.2, 0.8])

with col1:
    st.image(image_path, width=80)

with col2:
    st.title("Mobile Price Classification")

st.markdown("---")
st.write("Training Dataset Uploaded")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="uploaded_file")

if uploaded_file is not None:
    # Read and store data in session state only if it's a new file
    if st.session_state.data is None:
        st.session_state.data = pd.read_csv(uploaded_file)
    
    st.write("Top 5 rows of the dataset:")
    st.dataframe(st.session_state.data.head())

# Check if data exists when navigating back to this page
elif st.session_state.data is not None:
    st.write("Top 5 rows of the dataset (Persisted Data):")
    st.dataframe(st.session_state.data.head())

# Clear File Button
if st.button("Clear Uploaded File"):
    st.session_state.data = None  # Reset the stored data
    st.write("Uploaded file has been cleared.")
    st.rerun()
