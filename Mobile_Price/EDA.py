import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

st.title("Exploratory Data Analysis (EDA)")

st.markdown("---")

if "data" not in st.session_state or st.session_state.data is None:
    st.markdown("<p style='color:red; font-weight:bold;'>Please upload a CSV file in the Data page before proceeding with EDA.</p>", unsafe_allow_html=True)
    st.stop()

data = st.session_state.data

st.write("Top 5 rows of the Dataset")
st.dataframe(data.head())

# Display the shape of the dataset
st.write("Shape of the dataset:")
st.write(data.shape)

st.markdown("---")

# Display the total number of missing values
missing_values = data.isnull().sum().sum()
st.write(f"Total Missing Values: {missing_values}")

# Display the total number of duplicated rows
duplicated_values = data.duplicated().sum()
st.write(f"Total Duplicated Rows: {duplicated_values}")

st.markdown("---")

st.subheader("Total Number of Unique Values for Each Column")
unique_values = data.nunique()
unique_values_df = pd.DataFrame({
    'Variable': unique_values.index,
    'Total Number of Unique Values': unique_values.values
})
unique_values_df.index = range(1, len(unique_values_df) + 1)
st.dataframe(unique_values_df)

st.markdown("---")

# Button for Detect Variables with unique values < 5
if st.button("Detect Variables with Unique Values < 5"):

    unique_values = data.nunique()
    low_unique_vars = unique_values[unique_values < 5]

    if not low_unique_vars.empty:
        st.session_state.low_unique_vars = low_unique_vars
        st.subheader("Variables with Less Than 5 Unique Values")
        st.dataframe(low_unique_vars.rename("Unique Values").reset_index(name="Variable"))

        # Plot bar charts and pie charts for these variables
        for var in low_unique_vars.index:
            st.subheader(f"Bar Chart for {var}")
            fig, ax = plt.subplots(figsize=(6, 4))  
            data[var].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Bar Chart for {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Pie chart for each variable
            st.subheader(f"Pie Chart for {var}")
            fig, ax = plt.subplots(figsize=(6, 4))  
            data[var].value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Pie Chart for {var}")
            ax.set_ylabel("")  # Remove the y-axis label
            st.pyplot(fig)

    else:
        st.write("No variables with fewer than 5 unique values detected.")

st.markdown("---")
# Show a description of the dataset
st.subheader("Dataset Description:")
st.write(data.describe())

st.markdown("---")
