import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

image_path = "Phone_image.png"

col1, col2 = st.columns([0.2,0.8])

with col1:
    st.image(image_path, width=80)
    
with col2:
    st.title("Mobile Price Classification")

st.markdown("---")
st.write("Training Dataset Uploaded")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="uploaded_file")

# File Handling (only when file is uploaded)
if uploaded_file is not None:
    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv(uploaded_file)
    
    data = st.session_state.data
    
    st.write("Top 5 rows of the dataset:")
    st.dataframe(data.head())

# Clear File Button (to reset the file uploader)
if st.button("Clear Uploaded File"):
    # Clear session state and reset file uploader
    if "data" in st.session_state:
        del st.session_state["data"]  
    if "uploaded_file" in st.session_state:
        del st.session_state["uploaded_file"] 
    
    st.write("Uploaded file has been cleared.")
    st.rerun()
