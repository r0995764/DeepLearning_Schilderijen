import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETUP & DATA UITPAKKEN ---
print("Omgeving controleren...")
if os.path.exists("dataset.zip"):
    if not os.path.exists("processed"):
        print("Dataset uitpakken...")
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
else:
    print("⚠️ Geen dataset.zip gevonden (of processed bestaat al).")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = "processed/test"
MODEL_PATH = "transfer_model_vgg16.h5"

# --- 2. DATA LADEN ---
print("Testset laden...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False # Volgorde behouden!
)

# --- 3. MODEL LADEN (SIMPELE METHODE) ---
print(f"Volledig model laden van {MODEL_PATH}...")

if os.path.exists(MODEL_PATH):
    try:
        # Op de VSC werkt dit wel, omdat de versies kloppen met de training!
        model = load_model(MODEL_PATH)
        print("✅ Model succesvol geladen!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR bij laden model: {e}")
        exit(1)
else:
    print(f"❌ Bestand {MODEL_PATH} niet gevonden!")
    exit(1)

# --- 4. VOORSPELLEN ---
print("Voorspellingen genereren...")
y_pred_probs = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in test_ds], axis=0)
class_names = test_ds.class_names

# --- 5. RAPPORT MAKEN ---
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n--- CLASSIFICATION REPORT ---")
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# --- 6. MATRIX PLOTTEN ---
print("Confusion Matrix genereren...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Voorspelling')
plt.ylabel('Echt')
plt.title('Confusion Matrix (VGG16 Transfer Learning)')
plt.savefig('confusion_matrix.png')
print("Klaar! Check 'confusion_matrix.png' en 'classification_report.txt'")