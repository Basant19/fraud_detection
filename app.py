from flask import Flask, render_template, request
import pandas as pd
import sys
from src.pipeline.predict_pipeline import start_prediction_pipeline
from src.exception import CustomException
from src.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            expected_fields = [
                "TransactionAmount", "AnomalyScore", "Amount", "Age",
                "AccountBalance", "SuspiciousFlag", "Hour", "gap", "Category"
            ]
            form_data = {field: request.form.get(field) for field in expected_fields}

            # Log input
            logger.info(f"Received input: {form_data}")

            # Validate all fields filled
            if any(v is None or v == "" for v in form_data.values()):
                raise CustomException("Empty input fields detected", sys)

            df_input = pd.DataFrame([form_data])
            df_input = df_input.astype({
                "TransactionAmount": float,
                "AnomalyScore": float,
                "Amount": float,
                "Age": int,
                "AccountBalance": float,
                "SuspiciousFlag": int,
                "Hour": int,
                "gap": int,
                # Category stays string
            })

            artifacts = start_prediction_pipeline(df_input)

            # Interpret results
            prediction = "Fraud" if artifacts.predictions[0] == 1 else "Not Fraud"

            # Get both probabilities
            p_not_fraud, p_fraud = artifacts.probabilities[0]
            p_not_fraud = round(float(p_not_fraud), 3)
            p_fraud = round(float(p_fraud), 3)

            logger.info(f"Prediction: {prediction}, Prob(Not Fraud): {p_not_fraud}, Prob(Fraud): {p_fraud}")

            return render_template(
                "index.html",
                prediction=prediction,
                p_not_fraud=p_not_fraud,
                p_fraud=p_fraud
            )


  

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            error_message = CustomException(e, sys)
            return render_template("index.html", prediction=f"Error: {error_message}")

    return render_template("index.html")


if __name__ == "__main__":
    # Auto-open browser not possible natively, but host is fixed
    app.run(host="127.0.0.1", port=8080, debug=True)
