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
def build_model():

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


  # הצגת מבנה המודל
  model.summary()


  # Callbacks
  early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1)


  # שלב שני - Fine-Tuning
  base_model.trainable = True
  for layer in base_model.layers[:31]:  # מקפיאים את 100 השכבות הראשונות
      layer.trainable = False

  # קומפילציה מחדש עם Fine-Tuning
  model.compile(optimizer=Adam(learning_rate=0.00001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

def train_model_with_optimizer(optimizer, lr, epochs):
    model = build_model()

    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'sgd_momentum':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

   # early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-8, verbose=1)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights_dict,  # שימוש במשקולות למחלקות
        callbacks=[reduce_lr]#early_stop, reduce_lr]
    )


    return model, history

# הגדרת ערכים של Learning Rate ו-Epochs לבדוק
learning_rates = [0.01, 0.001]#, 0.001]  # דוגמאות לערכים שונים של Learning Rate
epochs_values = [ 10, 20]  # דוגמאות לערכים שונים של Epochs

# מילון לשמירת התוצאות
history_results = {}

# אופטימיזרים
optimizers = [ 'adam', 'rmsprop','sgd', 'sgd_momentum']

# אימון המודל

# לולאה עבור כל אופטימיזר, Learning Rate ו-Epochs
for opt in optimizers:
    history_results[opt] = {}
    for lr in learning_rates:
        for epochs in epochs_values:
            print(f"Training with {opt} optimizer, Learning Rate: {lr}, Epochs: {epochs}")
            model, history = train_model_with_optimizer(opt, lr, epochs)
            history_results[opt][(lr, epochs)] = history
            # הערכת המודל על סט הבדיקה
            test_loss, test_acc = model.evaluate(test_generator)
            print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

            # חישוב Precision, Recall, F1 ו-Accuracy
            # חיזוי על ה-Validation set
            y_probs = model.predict(test_generator)
            y_pred = (y_probs > 0.5).astype(int)
            y_true = test_generator.classes


            # חישוב והדפסת תוצאות לכל סף מ-0.1 עד 0.9
            thresholds = np.arange(0.1, 0.95, 0.05)
            precision_list = []
            recall_list = []

            plt.figure(figsize=(8, 6))

            for t in thresholds:
                y_pred = (y_probs > t).astype(int)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                precision_list.append(precision)
                recall_list.append(recall)

                print(f"Threshold {t:.2f} => Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            # ציור גרף Precision-Recall עם נקודות לכל סף
            plt.plot(recall_list, precision_list, marker='o', linestyle='-', color='b', label='Precision-Recall')

            # תוויות והגדרות לגרף
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision vs Recall for {opt} - LR: {lr}, Epochs: {epochs}')
            plt.legend()
            plt.grid(True)
            plt.show()
             # Loss
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{opt} - LR: {lr}, Epochs: {epochs} - Loss over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{opt} - LR: {lr}, Epochs: {epochs} - Accuracy over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.show()


