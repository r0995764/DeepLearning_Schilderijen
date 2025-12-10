import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. SETUP ---
# Pas dit straks aan naar de map op de VSC
# We gaan ervan uit dat je de zip uitpakt in dezelfde map
DATA_DIR = "processed" 

# Als de map nog niet bestaat, pak de zip uit (handig voor VSC)
if not os.path.exists(DATA_DIR) and os.path.exists("dataset.zip"):
    print("Zipping dataset uitpakken...")
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. DATA LADEN ---
print("Data laden...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Performance settings
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. MODEL BOUWEN ---
print("Model bouwen...")
num_classes = 4 # Rembrandt, Picasso, Mondriaan, Rubens

model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(224, 224, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. TRAINEN ---
print("Start training...")
# Op VSC kunnen we meer epochs doen!
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20 
)

# --- 5. OPSLAAN ---
print("Resultaten opslaan...")

# Model opslaan
model.save('baseline_model.keras')

# Grafiek opslaan als plaatje (ipv tonen op scherm)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('training_result.png')
print("Klaar! Check training_result.png en baseline_model.keras")