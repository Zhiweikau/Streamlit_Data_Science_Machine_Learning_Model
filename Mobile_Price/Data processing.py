import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.title("Data processing")

st.markdown("---")

if "data" not in st.session_state or st.session_state.data is None:
    st.markdown("<p style='color:red; font-weight:bold;'>Please upload a CSV file in the Data page before proceeding.</p>", unsafe_allow_html=True)
    st.stop()

data = st.session_state.data

target_column = 'price_range'
feature_columns = [col for col in data.columns if col != target_column]

x = data[feature_columns]
y = data[target_column]

# Train-test split
test_size = 0.2
random_state = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

# Add a selectbox to choose the scaler
normalization_method = st.selectbox(
    "Select the Normalization Method", 
    ["Select a Normalization Method", "StandardScaler", "MinMaxScaler"]
)

# Initialize variables to avoid NameError
x_train_scaled = None
x_test_scaled = None
scaler = None

if normalization_method == "StandardScaler":
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    st.write("StandardScaler applied on the data.")

elif normalization_method == "MinMaxScaler":
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    st.write("MinMaxScaler applied on the data.")

elif normalization_method == "Select a Normalization Method":
    st.write("<p style='color:red; font-weight:bold;'>Please Select a Normalization Method to Apply.</p>", unsafe_allow_html=True)

if x_train_scaled is not None and x_test_scaled is not None:
    st.session_state.x_train_scaled = x_train_scaled
    st.session_state.x_test_scaled = x_test_scaled
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    if scaler is not None:
        st.session_state.scaler = scaler

st.markdown("---")

# Displaying the size of the sets and the scaled version
st.write(f'Original set size: {data.shape[0]} samples')
st.write(f'Training set size: {x_train.shape[0]} samples')
st.write(f'Testing set size: {x_test.shape[0]} samples')

# Optionally show the scaled data (first few rows)
if st.checkbox("Show scaled training data"):
    st.write(pd.DataFrame(x_train_scaled, columns=feature_columns).head())
else:
    st.write("<p style='color:red; font-weight:bold;'>Please proceed to choose the Normalization Method first.</p>", unsafe_allow_html=True)
