# ui_safe.py
import streamlit as st
import pandas as pd

def safe_dataframe(df: pd.DataFrame, gradient: bool = True):
    try:
        if gradient:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                st.dataframe(df.style.background_gradient(subset=numeric_cols))
                return
        st.dataframe(df)
    except Exception:
        st.dataframe(df)
