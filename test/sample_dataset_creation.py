import pandas as pd

# Load sample of your training/test data
df = pd.read_csv(r"D:\fraud_detection\artifacts\raw_data\raw.csv")  # or raw test CSV
df_sample = df.head(5)  # just 5 rows for quick test
df_sample.to_csv("artifacts/sample_input.csv", index=False)
