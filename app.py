from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import lime
import lime.lime_tabular

app = Flask(__name__)

# =========================================================
# LOAD TRAINED MODEL FILES
# =========================================================

model = joblib.load("ffnn_model.pkl")
scaler = joblib.load("scaler.pkl")
X_train_scaled = joblib.load("X_train_scaled.pkl")

# =========================================================
# CREATE UPLOADS FOLDER
# =========================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# FEATURES USED DURING TRAINING
# =========================================================

FEATURES = [
    "rxbytes_rate",
    "txbytes_rate",
    "timecpu",
    "timesys",
    "timeusr",
    "state",
    "cputime",
    "memminor_fault",
    "memunused",
    "memlast_update",
    "memrss",
    "vdard_req_rate",
    "vdard_bytes_rate",
    "vdawr_reqs_rate",
    "vdawr_bytes_rate",
    "hdard_req_rate",
    "hdard_bytes_rate"
]

# =========================================================
# CONVERT TRAINING DATA TO NUMPY ARRAY
# =========================================================

X_train_scaled_np = np.array(X_train_scaled)

# =========================================================
# CREATE LIME EXPLAINER
# =========================================================

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled_np,
    feature_names=FEATURES,
    class_names=["Normal", "Attack"],
    mode="classification",
    discretize_continuous=True
)

# =========================================================
# LIME PREDICTION FUNCTION
# =========================================================

def lime_predict_fn(data):

    attack_probs = model.predict(data).flatten()

    normal_probs = 1 - attack_probs

    return np.column_stack((normal_probs, attack_probs))

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "upload.html",
        snapshot_index=1
    )

# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("dataset")

    # ---------------------------------------------
    # CHECK IF FILE EXISTS
    # ---------------------------------------------

    if file is None or file.filename == "":

        return render_template(
            "upload.html",
            error="No file selected. Please upload a CSV file.",
            snapshot_index=request.form.get("snapshot_index", 1)
        )

    # ---------------------------------------------
    # SAVE FILE
    # ---------------------------------------------

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    try:

        # -----------------------------------------
        # READ DATASET
        # -----------------------------------------

        df = pd.read_csv(filepath)

        # -----------------------------------------
        # GET SNAPSHOT NUMBER
        # -----------------------------------------

        snapshot_index = int(
            request.form.get("snapshot_index", 1)
        )

        row_index = snapshot_index - 1

        # -----------------------------------------
        # VALIDATE SNAPSHOT NUMBER
        # -----------------------------------------

        if row_index < 0 or row_index >= len(df):

            return render_template(
                "upload.html",
                error=f"Invalid snapshot number. Please enter a number between 1 and {len(df)}.",
                snapshot_index=snapshot_index
            )

        # -----------------------------------------
        # CHECK REQUIRED FEATURES
        # -----------------------------------------

        missing_cols = [
            col for col in FEATURES
            if col not in df.columns
        ]

        if missing_cols:

            return render_template(
                "upload.html",
                error=f"Missing required columns: {missing_cols}",
                snapshot_index=snapshot_index
            )

        # -----------------------------------------
        # SELECT VM SNAPSHOT
        # -----------------------------------------

        snapshot = df[FEATURES].iloc[[row_index]]

        # -----------------------------------------
        # SCALE SNAPSHOT
        # -----------------------------------------

        snapshot_scaled = scaler.transform(snapshot)

        # -----------------------------------------
        # MODEL PREDICTION
        # -----------------------------------------

        raw_prediction = model.predict(snapshot_scaled)

        probability = float(
            raw_prediction.flatten()[0]
        )

        # -----------------------------------------
        # CLASSIFICATION
        # -----------------------------------------

        if probability >= 0.5:

            prediction = "Attack"

            confidence = probability * 100

            attack_probability = probability * 100

            normal_probability = (
                1 - probability
            ) * 100

        else:

            prediction = "Normal"

            confidence = (
                1 - probability
            ) * 100

            attack_probability = (
                probability
            ) * 100

            normal_probability = (
                1 - probability
            ) * 100

        # =================================================
        # GENERATE LIME EXPLANATION
        # =================================================

        lime_exp = explainer.explain_instance(
            data_row=snapshot_scaled[0],
            predict_fn=lime_predict_fn,
            num_features=8
        )

        lime_features = lime_exp.as_list()

        # =================================================
        # SEND RESULTS TO HTML
        # =================================================

        return render_template(
            "upload.html",

            prediction=prediction,

            confidence=round(
                confidence,
                2
            ),

            normal_probability=round(
                normal_probability,
                2
            ),

            attack_probability=round(
                attack_probability,
                2
            ),

            snapshot_index=snapshot_index,

            lime_features=lime_features
        )

    # =====================================================
    # HANDLE ERRORS
    # =====================================================

    except Exception as e:

        return render_template(
            "upload.html",
            error=f"Error processing file: {str(e)}",
            snapshot_index=request.form.get(
                "snapshot_index",
                1
            )
        )

# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
