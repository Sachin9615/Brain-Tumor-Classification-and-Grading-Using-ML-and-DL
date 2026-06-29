import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

TRAIN_DIR  = r'C:\brain-tumor-project\dataset\Training'
TEST_DIR   = r'C:\brain-tumor-project\dataset\Testing'
MODEL_SAVE = r'C:\brain-tumor-project\brain_tumor_model.h5'

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
NUM_CLASSES = 4

# ─────────────────────────────────────────────
# DATA GENERATORS (STRONG AUGMENTATION)
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ✅ IMPORTANT: Check class mapping
print("\nCLASS INDICES:", train_gen.class_indices)

# ─────────────────────────────────────────────
# CLASS WEIGHTS (VERY IMPORTANT 🔥)
# ─────────────────────────────────────────────
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
print("\nCLASS WEIGHTS:", class_weights)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)   # 🔥 increased dropout
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

# ─────────────────────────────────────────────
# PHASE 1: TRAIN HEAD
# ─────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3, verbose=1)
]

print("\n🚀 Phase 1 Training...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# PHASE 2: FINE-TUNING
# ─────────────────────────────────────────────
base.trainable = True

# Freeze lower layers (VERY IMPORTANT)
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔧 Fine-tuning...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
print("\n📊 Evaluating...")
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")