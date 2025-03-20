import streamlit as st
from PIL import Image

from train.constants import STOCK_DATA_FOLDER

# Initialize session state for buttons
if "check" not in st.session_state:
    st.session_state.check = False
if "algo_comparison_button" not in st.session_state:
    st.session_state.algo_comparison_button = False
if "k_folds_button" not in st.session_state:
    st.session_state.k_folds_button = False

st.write(
    "This project helps you to predict the stock price for some big tech companies in the US"
)

ticker = st.selectbox(
    "Please choose the ticker of your company:",
    ["MSFT", "GOOGL", "IBM", "AMZN", "AAPL", "NVDA"],
)

if st.button("Predict the stock price!"):
    st.session_state.check = True


if st.session_state.check:
    ticker_folder_path = f"{STOCK_DATA_FOLDER}/{ticker}"

    back_test_image = Image.open(f"{ticker_folder_path}/{ticker}_backtest.png")
    st.title("Back test")
    st.image(back_test_image)

    if st.button("See the comparison chart for all the algorithms"):
        st.session_state.algo_comparison_button = True

    if st.session_state.algo_comparison_button:
        algo_comparison_image = Image.open(
            f"{ticker_folder_path}/{ticker}_algo_comparison.png"
        )
        st.title("Algorithms Comparison")
        st.image(algo_comparison_image)

    if st.button("See the chart for K Fold results"):
        st.session_state.k_folds_button = True

    if st.session_state.k_folds_button:
        k_folds_image = Image.open(f"{ticker_folder_path}/{ticker}_k_folds.png")
        st.title("K Fold Result")
        st.image(k_folds_image)
