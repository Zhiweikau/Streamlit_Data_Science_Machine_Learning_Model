import streamlit as st
import pandas as pd

st.title("Prediction for Price Range")
st.markdown("---")

if st.button("Clear the Selected Model"):
    # Remove model and related session state variables
    if 'model' in st.session_state:
        del st.session_state['model']
    if 'scaler' in st.session_state:
        del st.session_state['scaler']
    if 'x_train_scaled' in st.session_state:
        del st.session_state['x_train_scaled']
    if 'x_test_scaled' in st.session_state:
        del st.session_state['x_test_scaled']
    if 'best_model_name' in st.session_state:
        del st.session_state['best_model_name']
    if 'best_accuracy' in st.session_state:
        del st.session_state['best_accuracy']
    
    st.sidebar.success("Selected Model has been Cleaned.")

if "data" not in st.session_state or st.session_state.data is None:
    st.markdown("<p style='color:red; font-weight:bold;'>Please upload a CSV file in the Data to load the data first.</p>", unsafe_allow_html=True)
    st.stop()

data = st.session_state.data

if 'scaler' in st.session_state and 'best_model_name' in st.session_state and 'best_accuracy' in st.session_state:
    # Retrieve the best model and associated scaler from session state
    scaler = st.session_state.scaler  
    best_model_name = st.session_state.best_model_name
    best_accuracy = st.session_state.best_accuracy
    model = st.session_state.get('model', None)  # Default to None if not found

    # Display the best model name and accuracy
    st.markdown(f"<p style='color:yellow; font-weight:bold; font-size:30px;'>Selected Model: {best_model_name} (Accuracy: {best_accuracy:.2f})</p>", unsafe_allow_html=True)

    # File uploader for input data
    uploaded_file = st.file_uploader("Upload CSV file for Prediction", type="csv")
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(input_data)

        # Drop the target column if it exists
        input_features = input_data.drop(columns=['price_range'], errors='ignore')

        try:
            scaled_features = scaler.transform(input_features)
            scaled_features_df = pd.DataFrame(scaled_features, columns=input_features.columns)

            if "x_train_scaled" in st.session_state and "x_test_scaled" in st.session_state:
                x_train_scaled = st.session_state.x_train_scaled
                x_test_scaled = st.session_state.x_test_scaled
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
            
            # Check if the model has been trained and is available
            if model:
                model.fit(x_train_scaled, y_train)
                predictions = model.predict(scaled_features)

                # Append predictions to the data
                scaled_features_df['Predicted Price Range'] = predictions
                st.write("Predicted Results:")
                st.dataframe(scaled_features_df)

                # Option to download the predictions
                csv_data = scaled_features_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_data,
                    file_name='Prediction_Outcome.csv',
                    mime='text/csv'
                )
            else:
                st.write("<p style='color:red;'>No model is available for prediction.</p>", unsafe_allow_html=True)

        except Exception as e:
            st.write(f"<p style='color:red;'>Error in prediction: {str(e)}</p>", unsafe_allow_html=True)

    else:
        st.write("Please upload a CSV file to proceed.")
else:
    st.write("<p style='color:red; font-weight:bold;'>Please do the data processing > Train the Model first.</p>", unsafe_allow_html=True)
