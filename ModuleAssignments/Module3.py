import streamlit as st
import pandas as pd
import os

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Module 3 Application",
    layout="wide"
)

# TITLE
st.markdown(
    "<h1 style='text-align: center;'>Module 3 Application</h1>",
    unsafe_allow_html=True
)

# LOAD DATASET
base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir, "Salary_dataset.csv"))

# DESCRIPTION
st.subheader("Dataset Description")
st.text("Salary Dataset in CSV for Simple linear regression. It contains 30 rows, each having an entry number, number of years of experience, and a salary column.")

# COLUMN DROPDOWN
column = st.selectbox(
    "Select a Column",
    df.columns
)

# SLIDER FILTER
if df[column].dtype != "object":

    min_value = int(df[column].min())
    max_value = int(df[column].max())

    value_range = st.slider(
        "Select a Range",
        min_value,
        max_value,
        (min_value, max_value)
    )

    filtered_df = df[
        (df[column] >= value_range[0]) &
        (df[column] <= value_range[1])
    ]

else:
    filtered_df = df

# COLUMN SELECTION
st.subheader("Select Columns to Display")

selected_columns = []

cols = st.columns(len(df.columns))

for i, col_name in enumerate(df.columns):
    with cols[i]:
        if st.checkbox(col_name, value=True):
            selected_columns.append(col_name)

filtered_df = filtered_df[selected_columns]

# DISPLAY DATA
st.subheader("Dataset")
st.dataframe(filtered_df[selected_columns])

# SUMMARY STATISTICS
st.subheader("Summary Statistics")
numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
st.dataframe(numeric_df.describe())
