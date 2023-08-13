import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load the data from Excel files
labiba_data = pd.read_excel("data/labiba_200nagad.xlsx")
sakibul_data = pd.read_excel("data/sakibul_200nagad.xlsx")

# Ensure both datasets have the same length (200 data points)
if len(labiba_data) != len(sakibul_data):
    raise ValueError("Both datasets must have the same length")

# Extract the "Label" columns as lists
labiba_labels = labiba_data["Label"].tolist()
sakibul_labels = sakibul_data["Label"].tolist()

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labiba_labels, sakibul_labels)

print(f"Cohen's Kappa Score: {kappa_score:.4f}")
