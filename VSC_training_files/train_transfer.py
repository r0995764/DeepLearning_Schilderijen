import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# --- 1. SETUP ---
DATA_DIR = "processed" 
if not os.path.exists(DATA_DIR) and os.path.exists("dataset.zip"):
    print("Dataset uitpakken...")
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. DATA LADEN ---
print("Data laden...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/validation", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. MODEL BOUWEN (TRANSFER LEARNING) ---
print("VGG16 laden...")
# We laden VGG16 zonder de bovenkant. 
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = False # Eerst bevriezen!

inputs = tf.keras.Input(shape=(224, 224, 3))
# Augmentation laag
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.2)(x)
x = layers.RandomZoom(0.2)(x)

# VGG Preprocessing (BELANGRIJK!)
x = tf.keras.applications.vgg16.preprocess_input(x)

# Door VGG halen
x = conv_base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs, outputs)

# --- 4. FASE 1: FEATURE EXTRACTION ---
print("\n--- FASE 1: Feature Extraction ---")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Eerst 15 epochs grof trainen
history_1 = model.fit(train_ds, epochs=15, validation_data=val_ds)

# --- 5. FASE 2: FINE-TUNING ---
print("\n--- FASE 2: Fine-Tuning ---")
conv_base.trainable = True

# Zet alles op 'niet trainbaar', behalve het laatste blok (Block 5)
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Compileer opnieuw met hele lage learning rate (heel voorzichtig aanpassen)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Nog eens 15 epochs trainen (starten bij epoch 15)
history_2 = model.fit(train_ds, epochs=15, validation_data=val_ds)

# --- 6. RESULTATEN VERWERKEN ---
print("Grafieken genereren...")

# We doen EERST de grafieken. Als het opslaan van het model mislukt,
# hebben we tenminste de plaatjes nog voor het verslag!
acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
loss = history_1.history['loss'] + history_2.history['loss']
val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=14, color='green', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Accuracy (Transfer Learning)')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=14, color='green', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Loss (Transfer Learning)')

# Sla de grafiek op
plt.savefig('transfer_learning_result.png')
print("Grafiek opgeslagen!")

# --- 7. MODEL OPSLAAN ---
print("Model opslaan...")

# FIX: We gebruiken .h5 extensie, dat is stabieler dan .keras op oudere TF versies
try:
    model.save('transfer_model_vgg16.h5')
    print("Model succesvol opgeslagen als .h5")
except Exception as e:
    print(f"Kon model niet opslaan als geheel bestand: {e}")
    # Backup plan: alleen de gewichten opslaan (altijd veilig)
    model.save_weights('transfer_model_weights.h5')
    print("Backup: Alleen gewichten opgeslagen.")

print("\nEIND EVALUATIE OP TEST SET:")
try:
    model.evaluate(test_ds)
except:
    print("Kon niet evalueren, maar training is klaar.")
