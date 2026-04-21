import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import matplotlib.pyplot as plt

# הגדרת פרמטרים
img_size = 224  # ResNet50V2 מצפה ל-224x224
batch_size = 16
train_path = "/content/drive/MyDrive/deeplearning_proj_data/project_data/train"
val_path = "/content/drive/MyDrive/deeplearning_proj_data/project_data/val"
test_path = "/content/drive/MyDrive/deeplearning_proj_data/project_data/test"

# Augmentation לסט האימון
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# יצירת מחוללי תמונות (Generators)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=False
)


classes = train_generator.classes  # רשימת התוויות לכל דוגמה בסט האימון
class_labels = np.unique(classes)  # מציאת התוויות הייחודיות (0 ו-1 במקרה בינארי)

# חישוב המשקלות לכל מחלקה
class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)

# המרה למילון לשימוש באימון המודל
class_weights_dict = dict(enumerate(class_weights))

# טעינת מודל ResNet50V2 מאומן מראש (ללא שכבת הסיווג)
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # הקפאת כל השכבות

# בניית המודל באמצעות Sequential
model = Sequential([
    base_model,  # שכבת הבסיס של ResNet50V2
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # שכבת יציאה בינארית
])

# קומפילציה
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# הצגת מבנה המודל
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1)

# אימון המודל (שלב ראשון - הקפאת ResNet50V2)
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)
# הערכת המודל על סט הבדיקה
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# חיזוי תוצאות על סט הבדיקה
y_probs = model.predict(test_generator)
y_pred = (y_probs > 0.5).astype(int)
y_true = test_generator.classes

# חישוב Precision, Recall ו-F1 Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# ערכים שונים של סף (Threshold) בטווח 0.1 עד 0.9 בקפיצות של 0.05
thresholds = np.arange(0.1, 1.0, 0.05)

# יצירת מערכים לאחסון Precision, Recall ו-F1 Score
precisions = []
recalls = []
f1_scores = []

# חישוב Precision, Recall ו-F1 עבור כל סף
for threshold in thresholds:
    y_pred = (y_probs > threshold).astype(int)
    precision = precision_score(test_generator.classes, y_pred)
    recall = recall_score(test_generator.classes, y_pred)
    f1 = f1_score(test_generator.classes, y_pred)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# יצירת גרף Precision-Recall
plt.figure(figsize=(10, 6))
plt.plot(precisions, recalls, marker='o', label='Precision-Recall Curve', color='b')

# ציור נקודות F-Score
for i in range(len(thresholds)):
    plt.text(precisions[i], recalls[i], f'F1: {f1_scores[i]:.2f}', fontsize=9)

# הגדרת הגרף
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall for Different Thresholds')
plt.grid(True)
plt.legend()
plt.show()




# שלב שני - Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:31]:  # מקפיאים את 100 השכבות הראשונות
    layer.trainable = False

# קומפילציה מחדש עם Fine-Tuning
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# אימון המודל עם Fine-Tuning
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr],class_weight=class_weights_dict
)

# הערכת המודל על סט הבדיקה
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# חיזוי תוצאות על סט הבדיקה
y_probs = model.predict(test_generator)
y_pred = (y_probs > 0.5).astype(int)
y_true = test_generator.classes

# חישוב Precision, Recall ו-F1 Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ערכים שונים של סף (Threshold) בטווח 0.1 עד 0.9 בקפיצות של 0.05
thresholds = np.arange(0.1, 1.0, 0.05)

# יצירת מערכים לאחסון Precision, Recall ו-F1 Score
precisions = []
recalls = []
f1_scores = []

# חישוב Precision, Recall ו-F1 עבור כל סף
for threshold in thresholds:
    y_pred = (y_probs > threshold).astype(int)
    precision = precision_score(test_generator.classes, y_pred)
    recall = recall_score(test_generator.classes, y_pred)
    f1 = f1_score(test_generator.classes, y_pred)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# יצירת גרף Precision-Recall
plt.figure(figsize=(10, 6))
plt.plot(precisions, recalls, marker='o', label='Precision-Recall Curve', color='b')

# ציור נקודות F-Score
for i in range(len(thresholds)):
    plt.text(precisions[i], recalls[i], f'F1: {f1_scores[i]:.2f}', fontsize=9)

# הגדרת הגרף
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall for Different Thresholds')
plt.grid(True)
plt.legend()
plt.show()
# גרפים להערכת ביצועי המודל
def plot_learning_curve(history, fine_tune_history=None):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    if fine_tune_history:
        plt.plot(fine_tune_history.history['loss'], label='Train Loss (Fine-Tune)', linestyle='dashed', color='blue')
        plt.plot(fine_tune_history.history['val_loss'], label='Validation Loss (Fine-Tune)', linestyle='dashed', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    if fine_tune_history:
        plt.plot(fine_tune_history.history['accuracy'], label='Train Accuracy (Fine-Tune)', linestyle='dashed', color='blue')
        plt.plot(fine_tune_history.history['val_accuracy'], label='Validation Accuracy (Fine-Tune)', linestyle='dashed', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')

    plt.show()

# הצגת הגרפים
plot_learning_curve(history, history_fine)
