import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import pandas_profiling
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report

# lotties
import json
import requests
from streamlit_lottie import st_lottie


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title="ML APP - Try Yourself", layout="wide")

if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)


# ---------------------------------#
# Model building
def build_model(df):
    df = df.loc[:100]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    st.markdown("**1.2. Dataset dimension**")
    st.write("X")
    st.info(X.shape)
    st.write("Y")
    st.info(Y.shape)

    st.markdown("**1.3. Variable details**:")
    st.write("X variable (first 20 are shown)")
    st.info(list(X.columns[:20]))
    st.write("Y variable")
    st.info(Y.name)

    # Build lazy model
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=split_size, random_state=seed_number
    )
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

    st.subheader("2. Table of Model Performance")

    st.write("Training set")
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train, "training.csv"), unsafe_allow_html=True)

    st.write("Test set")
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test, "test.csv"), unsafe_allow_html=True)

    st.subheader("3. Plot of Model Performance (Test set)")

    with st.markdown("**R-squared**"):
        # Tall
        predictions_test["R-Squared"] = [
            0 if i < 0 else i for i in predictions_test["R-Squared"]
        ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(
            y=predictions_test.index, x="R-Squared", data=predictions_test
        )
        ax1.set(xlim=(0, 1))
    st.markdown(imagedownload(plt, "plot-r2-tall.pdf"), unsafe_allow_html=True)
    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
    ax1.set(ylim=(0, 1))
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, "plot-r2-wide.pdf"), unsafe_allow_html=True)

    with st.markdown("**RMSE (capped at 50)**"):
        # Tall
        predictions_test["RMSE"] = [
            50 if i > 50 else i for i in predictions_test["RMSE"]
        ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
    st.markdown(imagedownload(plt, "plot-rmse-tall.pdf"), unsafe_allow_html=True)
    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt, "plot-rmse-wide.pdf"), unsafe_allow_html=True)

    with st.markdown("**Calculation time**"):
        # Tall
        predictions_test["Time Taken"] = [
            0 if i < 0 else i for i in predictions_test["Time Taken"]
        ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(
            y=predictions_test.index, x="Time Taken", data=predictions_test
        )
    st.markdown(
        imagedownload(plt, "plot-calculation-time-tall.pdf"), unsafe_allow_html=True
    )
    # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(
        imagedownload(plt, "plot-calculation-time-wide.pdf"), unsafe_allow_html=True
    )


# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file1-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format="pdf", bbox_inches="tight")
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


lottie_code = load_lottiefile("lottiefiles/data_to_ai.json")

with st.sidebar:
    st_lottie(lottie_code, speed=1, loop=True, quality="low")
    st.title("Sales Forecast Analysis")
    choice = st.radio(
        "Navigation",
        ["Upload Data", "Modeling and Evaluating"],
    )

if choice == "Upload Data":
    st.title("Welcome to our Machine Learning Website")
    st.header("Let me explore your data")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)
        st.write(
            "Great job! Now move to the EDA step to understand your dataset:sunglasses:"
        )
        if st.button("EDA"):
            st.title("Exploratory Data Analysis")
            profile_df = df.profile_report()
            st_profile_report(profile_df)
    elif file is None:
        st.info(
            "If you don't know how our application running yet, please try our dataset below by clicking!"
        )
        if st.button("👉:red[Press] to use Example Dataset👈"):
            wine = load_wine()
            df = pd.DataFrame(
                data=np.c_[wine["data"], wine["target"]],
                columns=wine["feature_names"] + ["target"],
            )
            df.to_csv("dataset.csv", index=None)
            st.dataframe(df)
            profile_df = df.profile_report()
            st_profile_report(profile_df)


if choice == "Modeling and Evaluating":
    st.title("Let Build Your Model🖥️")
    option = st.selectbox(
        "What would you like to build?",
        ("Example with our dataset", "Your already upload file"),
    )
    st.subheader("Spliting your dataset")
    split_size = st.slider("Data split ratio (% for Training Set)", 10, 90, 80, 5)
    seed_number = st.slider("Set the random seed number", 1, 100, 42, 1)
    st.info("P/s: You can pass the split step if you want")
    if option == "Example with our dataset" and st.button(
        "👉:red[Press] to use ML's models after split data👈"
    ):
        # Diabetes dataset
        # diabetes = load_diabetes()
        # X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        # Y = pd.Series(diabetes.target, name='response')
        # df = pd.concat( [X,Y], axis=1 )
        # st.markdown('The Diabetes dataset is used as the example.')
        # st.write(df.head(5))
        # Boston housing dataset
        wine = load_wine()
        # X = pd.DataFrame(wine.data, columns=wine.feature_names)
        # Y = pd.Series(wine.target, name='response')
        X = pd.DataFrame(wine.data, columns=wine.feature_names).loc[
            :100
        ]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        Y = pd.Series(wine.target, name="response").loc[
            :100
        ]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        df = pd.concat([X, Y], axis=1)
        st.markdown("The UCI ML Wine dataset is used as the example.")
        st.write(df.head(5))
        build_model(df)
    elif option == "Your already upload file":
        build_model(df)

# ----------
# -----------------------#

# ---------------------------------#
# Sidebar - Collects user input features into dataframe
# with st.sidebar.header("1. Upload your CSV data"):
#     uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# # ---------------------------------#
# # Main panel

# # Displays the dataset
# st.subheader("1. Dataset")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.markdown("**1.1. Glimpse of dataset**")
#     st.write(df)
#     build_model(df)
# else:
#     st.info("Awaiting for CSV file to be uploaded.")
#     if st.button("Press to use Example Dataset"):
#         # Diabetes dataset
#         # diabetes = load_diabetes()
#         # X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#         # Y = pd.Series(diabetes.target, name='response')
#         # df = pd.concat( [X,Y], axis=1 )

#         # st.markdown('The Diabetes dataset is used as the example.')
#         # st.write(df.head(5))

#         # Boston housing dataset
#         wine = load_wine()
#         # X = pd.DataFrame(wine.data, columns=wine.feature_names)
#         # Y = pd.Series(wine.target, name='response')
#         X = pd.DataFrame(wine.data, columns=wine.feature_names).loc[
#             :100
#         ]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
#         Y = pd.Series(wine.target, name="response").loc[
#             :100
#         ]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
#         df = pd.concat([X, Y], axis=1)

#         st.markdown("The Boston housing dataset is used as the example.")
#         st.write(df.head(5))

#         build_model(df)
