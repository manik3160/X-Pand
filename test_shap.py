import pickle
import numpy as np
import time

print("Loading explainer...")
with open("models/shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

X = np.random.rand(12000, 10)
print("Computing SHAP for 12,000 rows...")
start = time.time()
explainer.shap_values(X)
print(f"Time taken: {time.time() - start:.2f}s")
