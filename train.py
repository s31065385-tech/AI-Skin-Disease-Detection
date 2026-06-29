import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATASET_PATH = "dataset"
CSV_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")

df = pd.read_csv(CSV_PATH)

image_dir1 = os.path.join(DATASET_PATH, "HAM10000_images_part_1")
image_dir2 = os.path.join(DATASET_PATH, "HAM10000_images_part_2")

images = []
labels = []

label_map = {label: idx for idx, label in enumerate(df['dx'].unique())}
print(label_map)

for _, row in df.iterrows():
    image_name = row['image_id'] + ".jpg"

    img_path = os.path.join(image_dir1, image_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(image_dir2, image_name)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        images.append(img)
        labels.append(label_map[row['dx']])

X = np.array(images) / 255.0
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

model.save("models/skin_disease_model.h5")

print("Training Complete!")