import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# הגדרת פרמטרים
img_size = 160
batch_size = 10
train_path = "/content/drive/MyDrive/deeplearning_proj_data/project_data/train"
val_path = "/content/drive/MyDrive/deeplearning_proj_data/project_data/val"
test_path = "/content/drive/MyDrive/deeplearning_proj_data/project_data/test"

# Data Augmentation עבור אימון
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

# יצירת מחוללי תמונות (Generators) עם class_mode='categorical'
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',  # שונה ל-categorical
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# חישוב משקולות למחלקות (Class Weights)
classes = train_generator.classes
class_labels = np.unique(classes)
class_weights = compute_class_weight('balanced', classes=class_labels, y=classes)
class_weights_dict = dict(enumerate(class_weights))

# יצירת מודל CNN עם 3 קטגוריות ו-Softmax
def build_medical_cnn(input_shape=(img_size, img_size, 1), num_classes=3):
    model = Sequential()

    # שכבת קונבולוציה ראשונה
    model.add(Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))

    # שכבת קונבולוציה שניה
    model.add(Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))

    # שכבת קונבולוציה שלישית
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.4))

    # שכבת קונבולוציה רביעית
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    # שכבת קונבולוציה חמישית
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))

    # Flatten ו-Dense
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # שכבת הפלט עם Softmax ל-3 קטגוריות
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',  # שונה ל-categorical_crossentropy
                  metrics=['accuracy'])

    return model

# פונקציה לאימון המודל עם פרמטרים שונים
def train_model_with_optimizer(optimizer, lr, epochs, num_classes=3):
    model = build_medical_cnn(num_classes=num_classes)

    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'sgd_momentum':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-8, verbose=1)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights_dict,
        callbacks=[reduce_lr]
    )
    return model, history

# הגדרת ערכים של Learning Rate ו-Epochs לבדוק
learning_rates = [0.001, 0.0001]
epochs_values = [15 , 25]

# מילון לשמירת התוצאות
history_results = {}

# אופטימיזרים
optimizers = ['adam','sgd_momentum','rmsprop','sgd']

# לולאה עבור כל אופטימיזר, Learning Rate ו-Epochs
for opt in optimizers:
    history_results[opt] = {}
    for lr in learning_rates:
        for epochs in epochs_values:
            print(f"Training with {opt} optimizer, Learning Rate: {lr}, Epochs: {epochs}")
            model, history = train_model_with_optimizer(opt, lr, epochs)

            # הערכת המודל על סט הבדיקה
            test_loss, test_acc = model.evaluate(test_generator)
            print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

            # חיזוי על ה-Validation set
            y_probs = model.predict(test_generator)
            y_pred = np.argmax(y_probs, axis=1)  # בוחרים את הקטגוריה עם ההסתברות הגבוהה ביותר
            y_true = test_generator.classes

            # חישוב Precision, Recall, F1 לכל קטגוריה
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            cm = confusion_matrix(y_true, y_pred)
            # הצגת ה-Confusion Matrix בצורה גרפית
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()
                        # ציור גרפים של Train Loss, Validation Loss, Train Accuracy, Validation Accuracy
            plt.figure(figsize=(12, 6))

            # גרף Loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # גרף Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.show()

