import streamlit as st
import pandas as pd
import pickle as pkl
from pathlib import Path

from pycaret.classification import setup as setup_clf, compare_models as compare_clfs, evaluate_model as evaluate_clfs, pull as pull_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, evaluate_model as evaluate_reg, pull as pull_reg

# Page Configuration
st.set_page_config(page_title="AutoML Tool", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    body {
        background-color: #e6f2ff;
        font-family: 'Verdana', sans-serif;
        color: #333;
    }
    .main-title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 16px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Title
st.markdown("<div class='main-title'>AutoML: Automated Machine Learning</div>", unsafe_allow_html=True)
st.write("Effortlessly ingest data, select models, and download trained models.")

# Sidebar for Navigation
task = st.sidebar.radio("Select Task", ["Data Ingestion", "Model Selection", "Downloading Model"])

# Data Ingestion Section
if task == "Data Ingestion":
    st.markdown("<div class='section-header'>Upload Your Dataset</div>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview of Uploaded Data")
        st.dataframe(df.head())
        df.to_csv("sourcefile.csv", index=False)
        st.success("File successfully saved!")

# Model Selection Section
elif task == "Model Selection":
    st.markdown("<div class='section-header'>Select and Train a Model</div>", unsafe_allow_html=True)
    try:
        file = pd.read_csv("sourcefile.csv")
        task_type = st.radio("Choose Task Type", ["Regression", "Classification"])
        target = st.selectbox("Select Target Column", file.columns)
        
        if st.button("Run Model Selection"):
            if task_type == "Regression":
                setup_df = setup_reg(file, target=target)
                st.write("Setup Summary")
                st.dataframe(pull_reg())
                
                st.write("Comparing Models")
                best_model = compare_reg()
                st.dataframe(pull_reg())
                
                st.write("Evaluating Best Model")
                evaluate_reg(best_model)
                
                if st.button("Download Model"):
                    with open("model.pkl", "wb") as f:
                        pkl.dump(best_model, f)
                    st.success("Model saved as model.pkl!")
            
            elif task_type == "Classification":
                setup_df = setup_clf(file, target=target)
                st.write("Setup Summary")
                st.dataframe(pull_clf())
                
                st.write("Comparing Models")
                best_model = compare_clfs()
                st.dataframe(pull_clf())
                
                st.write("Evaluating Best Model")
                evaluate_clfs(best_model)
                
                if st.button("Download Model"):
                    with open("model.pkl", "wb") as f:
                        pkl.dump(best_model, f)
                    st.success("Model saved as model.pkl!")
    except FileNotFoundError:
        st.error("No data file found! Please upload a dataset in 'Data Ingestion' first.")

# Downloading Model Section
elif task == "Downloading Model":
    st.markdown("<div class='section-header'>Download Your Trained Model</div>", unsafe_allow_html=True)
    try:
        with open("model.pkl", "rb") as f:
            st.download_button(label="Download Trained Model", data=f, file_name="model.pkl", mime="application/octet-stream")
    except FileNotFoundError:
        st.warning("No model found! Train a model first in 'Model Selection'.")

# Footer
st.markdown("<div class='footer'>Developed by D.Harshitha</div>", unsafe_allow_html=True)
