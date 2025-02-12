import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt

st.title("Machine Learning Model Performance")
st.markdown("---")

if "data" not in st.session_state or st.session_state.data is None:
    st.markdown("<p style='color:red; font-weight:bold;'>Please upload a CSV file in the Data to load the data first.</p>", unsafe_allow_html=True)
    st.stop()

data = st.session_state.data

if "x_train_scaled" in st.session_state and "x_test_scaled" in st.session_state:
    x_train_scaled = st.session_state.x_train_scaled
    x_test_scaled = st.session_state.x_test_scaled
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # Binarize the target labels for multiclass ROC-AUC calculation
    classes = sorted(y_train.unique())
    y_train_binarized = label_binarize(y_train, classes=classes)
    y_test_binarized = label_binarize(y_test, classes=classes)

    # Create the selectbox options
    model_options = ["Please Select the Model", "Logistic Regression", "Decision Tree", "SVM", "Random Forest", "XGBoost", "Select All"]
    selected_model = st.selectbox("Select a Machine Learning Model", model_options)

    # Function to train and evaluate a model
    def evaluate_model(model, model_name):
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        y_pred_proba = model.predict_proba(x_test_scaled) if hasattr(model, "predict_proba") else None
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.markdown(f"<p style='color:yellow; font-size:30px; font-weight:bold;'>Performance of {model_name}</p>", unsafe_allow_html=True)
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.markdown("### Classification Report")
        st.dataframe(pd.DataFrame(report))

        # Plot ROC-AUC curve (one-vs-rest for multiclass)
        if y_pred_proba is not None:
            st.markdown("### ROC-AUC Curve (Multiclasses)")
            plt.figure(figsize=(10, 8))

            if y_test_binarized.ndim == 2:  # Multiclass case
                for i, class_name in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")
            else:  # Binary classification case
                fpr, tpr, _ = roc_curve(y_test_binarized, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")

            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC-AUC Curve: {model_name}")
            plt.legend(loc="lower right")
            st.pyplot(plt)

        return accuracy

    # Model selection logic
    if selected_model == "Logistic Regression":
        accuracy = evaluate_model(LogisticRegression(random_state=42), "Logistic Regression")
        st.session_state.model = LogisticRegression(random_state=42)
        st.session_state.best_model_name = "Logistic Regression"
        st.session_state.best_accuracy = accuracy

    elif selected_model == "Decision Tree":
        accuracy = evaluate_model(DecisionTreeClassifier(random_state=42), "Decision Tree")
        st.session_state.model = DecisionTreeClassifier(random_state=42)
        st.session_state.best_model_name = "Decision Tree"
        st.session_state.best_accuracy = accuracy

    elif selected_model == "SVM":
        accuracy = evaluate_model(SVC(random_state=42, probability=True), "Support Vector Machine")
        st.session_state.model = SVC(random_state=42, probability=True)
        st.session_state.best_model_name = "Support Vector Machine"
        st.session_state.best_accuracy = accuracy

    elif selected_model == "Random Forest":
        accuracy = evaluate_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
        st.session_state.model = RandomForestClassifier(n_estimators=100, random_state=42)
        st.session_state.best_model_name = "Random Forest"
        st.session_state.best_accuracy = accuracy

    elif selected_model == "XGBoost":
        accuracy = evaluate_model(XGBClassifier(random_state=42, eval_metric='logloss'), "XGBoost")
        st.session_state.model = XGBClassifier(random_state=42, eval_metric='logloss')
        st.session_state.best_model_name = "XGBoost"
        st.session_state.best_accuracy = accuracy

    elif selected_model == "Select All":
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(random_state=42, probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        }

        best_accuracy = 0
        best_model = None
        best_model_name = None

        # Loop through all models to find the best one based on accuracy
        for model_name, model in models.items():
            accuracy = evaluate_model(model, model_name)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

        # Save the best model in session state
        st.session_state.model = best_model
        st.session_state.best_model_name = best_model_name
        st.session_state.best_accuracy = best_accuracy

        # Display the best model
        st.markdown(f"<p style='color:green; font-size:20px;'>Best Model Based on Accuracy: {best_model_name} (Accuracy = {best_accuracy:.2f})</p>", unsafe_allow_html=True)

    elif selected_model == "Please Select the Model":
        st.write("<p style='color:red; font-weight:bold;'>Please select a model to view its performance.</p>", unsafe_allow_html=True)

else:
    st.write("<p style='color:red; font-weight:bold;'>Please make sure this step is after Data Processing.</p>", unsafe_allow_html=True)
