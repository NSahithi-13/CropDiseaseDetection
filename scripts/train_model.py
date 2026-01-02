# ---------------- Step 1: Load and preprocess images ----------------
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset_path = "D:/Sahithi/CropDiseaseDetection/dataset/PlantVillage"
image_size = 64

X = []
y = []
classes = os.listdir(dataset_path)
classes.sort()
print("Classes:", classes)

for idx, cls in enumerate(classes):
    cls_path = os.path.join(dataset_path, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print("Skipped (cannot read):", img_path)
            continue
        img = cv2.resize(img, (image_size, image_size))
        img = img / 255.0
        X.append(img)
        y.append(idx)


X = np.array(X)
y = to_categorical(y, num_classes=len(classes))
print("Data shape:", X.shape, y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ---------------- Step 2: Build CNN and train ----------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

model.save("models/crop_disease_model.h5")
print("Model saved!")
