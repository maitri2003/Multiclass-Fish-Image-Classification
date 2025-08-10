# pick_best_model.py
import pandas as pd
import shutil, os

df = pd.read_csv("model_evaluation_results.csv")
# sort by Accuracy (or "F1-score")
best = df.sort_values(by="Accuracy", ascending=False).iloc[0]
best_model = best["Model"]

SRC = os.path.join("..", "models", best_model)
DST = os.path.join("..", "models", "final_model.h5")
shutil.copyfile(SRC, DST)
print(f"Picked best model: {best_model} -> saved as final_model.h5")
