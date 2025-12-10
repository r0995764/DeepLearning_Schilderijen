import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 1. SETUP ---
# We gaan ervan uit dat de dataset al uitgepakt is door je vorige job.
# Zo niet, pakken we hem weer uit.
DATA_DIR = "processed" 
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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. DATA AUGMENTATION ---
# Dit is NIEUW! We definiëren bewerkingen die we op de plaatjes loslaten.
data_augmentation = models.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1), # Draai tot 10%
  layers.RandomZoom(0.1),     # Zoom tot 10%
])

# --- 4. MODEL MET AUGMENTATION ---
print("Model bouwen met Augmentation...")
num_classes = 4 

model = models.Sequential([
  layers.Input(shape=(224, 224, 3)),
  
  # Hier voegen we de augmentation toe. 
  # Dit is alleen actief tijdens het trainen, niet tijdens het testen!
  data_augmentation,
  
  layers.Rescaling(1./255),
  
  # Zelfde lagen als de baseline
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  # Een Dropout laag helpt ook tegen overfitting (willekeurig neuronen uitzetten)
  layers.Dropout(0.5), 
  layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. TRAINEN ---
print("Start training...")
# Met augmentation duurt het leren langer, dus we doen 30 epochs ipv 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=30 
)

# --- 6. OPSLAAN ---
print("Resultaten opslaan...")
model.save('augmented_model.keras')

# Grafieken maken
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
plt.title('Training and Validation Accuracy (Augmented)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (Augmented)')

plt.savefig('augmented_result.png')
print("Klaar!")