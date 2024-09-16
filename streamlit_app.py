import streamlit as st
import os
from PIL import Image

st.write("This project helps you to predict the stock price for some big tech companies in the US")

ticker = st.selectbox("Please choose the ticker of your company:", ["MSFT", "GOOGL", "IBM", "AMZN", "AAPL", "NVDA"])

check = st.button("Predict the stock price!")

if check:
    dir_path = os.path.dirname(os.path.abspath(__file__))

    algo_comparison_image = Image.open(f"{dir_path}/{ticker}/{ticker}_algo_comparison.png")
    st.title("Algorithms Comparison")
    st.image(algo_comparison_image)

    k_folds_image = Image.open(f"{dir_path}/{ticker}/{ticker}_k_folds.png")
    st.title("K Fold Result")
    st.image(k_folds_image)
    
    back_test_image = Image.open(f"{dir_path}/{ticker}/{ticker}_backtest.png")
    st.title("Back test")
    st.image(back_test_image)
