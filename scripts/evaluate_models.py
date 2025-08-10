# evaluate_models.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# ✅ Paths
MODELS_DIR = "../models"
TEST_DIR = "../data/test"

# ✅ Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Load Test Data
test_gen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ✅ Store results
results = []

# ✅ Loop over all .h5 models in models folder
for model_file in os.listdir(MODELS_DIR):
    if model_file.endswith(".h5"):
        model_path = os.path.join(MODELS_DIR, model_file)
        print(f"\n🔍 Evaluating model: {model_file}")

        try:
            # Load and compile the model
            model = load_model(model_path)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # Evaluate on test set
            loss, accuracy = model.evaluate(test_data, verbose=0)

            # Predictions for precision/recall/f1
            y_pred = model.predict(test_data, verbose=0)
            y_pred_classes = y_pred.argmax(axis=1)
            y_true = test_data.classes

            report = classification_report(y_true, y_pred_classes, target_names=test_data.class_indices.keys(), output_dict=True)

            results.append({
                "Model": model_file,
                "Accuracy": accuracy,
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1-score": report["weighted avg"]["f1-score"]
            })

        except Exception as e:
            print(f"❌ Failed to evaluate {model_file}: {e}")

# ✅ Convert results to DataFrame
df = pd.DataFrame(results)
print("\n📊 Evaluation Results:")
print(df)

# Save results as CSV
df.to_csv("model_evaluation_results.csv", index=False)

# ✅ Plot comparison
if not df.empty:
    df.plot(x="Model", y=["Accuracy", "Precision", "Recall", "F1-score"], kind="bar", figsize=(10, 6))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()
else:
    print("⚠️ No models were evaluated.")
