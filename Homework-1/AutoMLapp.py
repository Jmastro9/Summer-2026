import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# PAGE CONFIG
st.set_page_config(
    page_title="AutoML Regression application",
    layout="wide"
)

#CSS STYLINGS
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Centered title */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    margin-bottom: 40px;
    color: white;
}

/* Styled containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #161b22;
    border-radius: 12px;
    padding-left: 10px;
    padding-right: 10px;
}

/* Remove double border */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown(
    "<div class='title'>AutoML application</div>",
    unsafe_allow_html=True
)

# MAIN
left_col, middle_col, right_col = st.columns([1.3, 1, 2])
df = None

# LEFT COLUMN
with left_col:
    with st.container(border=True):

        st.subheader("Upload Tabular Data")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("")

    with st.container(border=True):

        st.subheader("File Preview")

        if df is not None:
            st.dataframe(
                df.head(10),
                use_container_width=True,
                height=300
            )
        else:
            st.info("Upload a dataset to preview.")

# MIDDLE COLUMN
with middle_col:
    with st.container(border=True):

        st.subheader("Model Selection")
        st.write("Deselect models to exclude them:")

        linear_regression = st.checkbox("Linear Regression", value=True)
        random_forest = st.checkbox("Random Forest Regression", value=True)
        neural_network = st.checkbox("Neural Network", value=True)

# RIGHT COLUMN
with right_col:
    with st.container(border=True):

        st.subheader("Results")

        if df is None:
            st.info("Upload a dataset to see results.")
        elif not any([linear_regression, random_forest, neural_network]):
            st.warning("Please select at least one model.")
        elif df.shape[1] < 2:
            st.error(
                "Dataset must contain at least "
                "one feature column and one target column."
            )
        else:
            try:

                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                X = pd.get_dummies(X)
                X = X.fillna(0)

                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42
                )

                selected_models = []

                if linear_regression:
                    selected_models.append("Linear Regression")
                if random_forest:
                    selected_models.append("Random Forest")
                if neural_network:
                    selected_models.append("Neural Network")

                progress_bar = st.progress(0)
                status = st.empty()

                results = []

                for i, model_name in enumerate(selected_models):

                    status.write(f"Running {model_name}...")
                    time.sleep(1)

                    if model_name == "Linear Regression":

                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        r2 = r2_score(y_test, predictions)
                        rmse = root_mean_squared_error(y_test, predictions)

                        results.append({
                            "name": "Linear Regression",
                            "r2": r2,
                            "rmse": rmse,
                            "predictions": predictions.tolist(),
                            "actuals": y_test.tolist()
                        })

                    elif model_name == "Random Forest":

                        model = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        r2 = r2_score(y_test, predictions)
                        rmse = root_mean_squared_error(y_test, predictions)

                        results.append({
                            "name": "Random Forest Regression",
                            "r2": r2,
                            "rmse": rmse,
                            "predictions": predictions.tolist(),
                            "actuals": y_test.tolist()
                        })

                    elif model_name == "Neural Network":

                        model = MLPRegressor(
                            hidden_layer_sizes=(100,),
                            max_iter=500,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        r2 = r2_score(y_test, predictions)
                        rmse = root_mean_squared_error(y_test, predictions)

                        results.append({
                            "name": "Neural Network Regression",
                            "r2": r2,
                            "rmse": rmse,
                            "predictions": predictions.tolist(),
                            "actuals": y_test.tolist()
                        })

                    progress_bar.progress((i + 1) / len(selected_models))

                # DISPLAY RESULTS
                status.success("All selected models completed.")

                tabs = st.tabs([result["name"] for result in results])

                for tab, result in zip(tabs, results):
                    with tab:

                        r2_percent = result["r2"] * 100

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R² Score", f"{r2_percent:.2f}%")
                        with col2:
                            st.metric("RMSE", f"{result['rmse']:.2f}")

                        if result["r2"] >= 0.90:
                            st.success("Excellent fit")
                        elif result["r2"] >= 0.70:
                            st.info("Good fit")
                        elif result["r2"] >= 0.50:
                            st.warning("Moderate fit")
                        else:
                            st.error("Poor fit")
    
                        preds = result["predictions"]
                        actuals = result["actuals"]

                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.scatter(preds, actuals, alpha=0.6, edgecolors="none",
                                   color="#378ADD", s=30, label="Predictions")

                        combined_min = min(min(preds), min(actuals))
                        combined_max = max(max(preds), max(actuals))
                        ax.plot(
                            [combined_min, combined_max],
                            [combined_min, combined_max],
                            color="#E24B4A", linewidth=1.5,
                            linestyle="--", label="Perfect fit"
                        )

                        ax.set_xlabel("Predicted", fontsize=11, color="#888780")
                        ax.set_ylabel("Actual", fontsize=11, color="#888780")
                        ax.set_facecolor("#161b22")
                        fig.patch.set_facecolor("#161b22")
                        ax.tick_params(colors="#888780")
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#333a45")
                        ax.legend(fontsize=9, facecolor="#161b22", labelcolor="white",
                                  edgecolor="#333a45")

                        st.pyplot(fig)
                        plt.close(fig)

            except Exception as e:
                st.error(f"Model execution failed: {e}")

# FOOTER
st.markdown("---")