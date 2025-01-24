import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Chart")

st.markdown("---")

# Callback the session_state
if "data" in st.session_state:
    data = st.session_state.data

    if "low_unique_vars" in st.session_state:
        low_unique_vars = st.session_state.low_unique_vars

        # Define the function to get numerical columns
        def get_numerical_columns(data):
            return data.select_dtypes(include=[np.number]).columns.tolist()

        # Create a selectbox for chart types
        chart_type = st.selectbox("Select a Chart Type", ["Select a Chart", "Box Plot", "Scatter Plot", "Histogram", "Density Plot", "Correlation Heatmap"])

        # Box Plot
        if chart_type == "Box Plot":
            selected_var = st.selectbox("Select Variable for X-axis", ["Select a Variable"] + list(low_unique_vars.index))

            if selected_var != "Select a Variable":
                st.subheader(f"Box Plot: {selected_var} vs. All Numerical Variables")

                numerical_columns = get_numerical_columns(data)

                # Display a progress bar for large datasets
                progress_bar = st.progress(0)
                num_plots = len(numerical_columns)

                # Plotting box plots for all numerical columns with selected variable as x-axis
                fig, axes = plt.subplots(len(numerical_columns), 1, figsize=(10, 6 * num_plots))
                if num_plots == 1:
                    axes = [axes]

                for i, col in enumerate(numerical_columns):
                    sns.boxplot(x=selected_var, y=col, data=data, ax=axes[i])
                    axes[i].set_title(f"{selected_var} vs {col}")
                    axes[i].set_xlabel(selected_var)
                    axes[i].set_ylabel(col)

                    # Update progress bar
                    progress_bar.progress((i + 1) / num_plots)

                # After plotting, clear the progress bar
                progress_bar.empty()

                st.pyplot(fig)

        # Scatter Plot
        elif chart_type == "Scatter Plot":
            selected_var = st.selectbox("Select Variable for X-axis", ["Select a Variable"] + list(data.columns))

            if selected_var != "Select a Variable":
                selected_var_scatter = st.selectbox("Select Another Variable for Y-axis", ["Select a Variable"] + list(get_numerical_columns(data)))

                if selected_var_scatter != "Select a Variable" and selected_var_scatter != selected_var:
                    st.subheader(f"Scatter Plot: {selected_var} vs {selected_var_scatter}")

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(data[selected_var], data[selected_var_scatter], alpha=0.5, color='orange')
                    ax.set_title(f"Scatter Plot of {selected_var} vs {selected_var_scatter}")
                    ax.set_xlabel(selected_var)
                    ax.set_ylabel(selected_var_scatter)
                    st.pyplot(fig)

        # Histogram
        elif chart_type == "Histogram":
            numerical_columns = get_numerical_columns(data)
            selected_var_hist = st.selectbox("Select Variable for Histogram", ["Select a Variable"] + numerical_columns)

            if selected_var_hist != "Select a Variable":
                st.subheader(f"Histogram of {selected_var_hist}")
                fig, ax = plt.subplots(figsize=(8, 6))
                data[selected_var_hist].plot(kind='hist', ax=ax, bins=30, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {selected_var_hist}")
                ax.set_xlabel(selected_var_hist)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        # Density Plot (KDE)
        elif chart_type == "Density Plot":
            numerical_columns = get_numerical_columns(data)
            selected_var_kde = st.selectbox("Select Variable for Density Plot", ["Select a Variable"] + numerical_columns)

            if selected_var_kde != "Select a Variable":
                st.subheader(f"Density Plot of {selected_var_kde}")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.kdeplot(data[selected_var_kde], ax=ax, fill=True, color='green')
                ax.set_title(f"Density Plot of {selected_var_kde}")
                ax.set_xlabel(selected_var_kde)
                ax.set_ylabel("Density")
                st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":
            corr_matrix = data.corr()

            # Plot the heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlation Matrix Heatmap')
            st.pyplot(plt)

    else:
        st.write("<p style='color:red; font-weight:bold;'>Please Go to EDA page to press the Button 'Detect Variables'.</p>",unsafe_allow_html=True)

else:
    st.markdown("<p style='color:red; font-weight:bold;'>Please upload a CSV file in the Data to load the data first.</p>", unsafe_allow_html=True)
