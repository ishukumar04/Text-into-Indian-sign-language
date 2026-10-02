import os
import cv2
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "dataset"
MODEL_FOLDER = "model"

MODEL_PATH = "model/isl_model.h5"
CLASSES_PATH = "model/classes.txt"


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)


# =========================================================
# GET CLASSES
# =========================================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])


print()
print("Classes:")
print(classes)
print()


if len(classes) < 2:
    print("ERROR: At least two classes are required.")
    exit()


# =========================================================
# LANDMARK EXTRACTION
# =========================================================

def extract_landmarks(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(image_rgb)

    # -----------------------------------------
    # No hands detected
    # -----------------------------------------

    if not result.multi_hand_landmarks:

        return None


    # -----------------------------------------
    # Create two empty hands
    # -----------------------------------------

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)


    # -----------------------------------------
    # Extract detected hands
    # -----------------------------------------

    if result.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):

            landmarks = []

            for point in hand_landmarks.landmark:

                landmarks.append(point.x)
                landmarks.append(point.y)
                landmarks.append(point.z)


            landmarks = np.array(
                landmarks,
                dtype=np.float32
            )


            # ---------------------------------
            # Determine left / right hand
            # ---------------------------------

            label = handedness.classification[0].label


            if label == "Left":

                left_hand = landmarks

            elif label == "Right":

                right_hand = landmarks


    # -----------------------------------------
    # Combine
    # -----------------------------------------

    combined = np.concatenate([
        left_hand,
        right_hand
    ])


    return combined


# =========================================================
# CREATE DATASET
# =========================================================

X = []
y = []


for class_index, class_name in enumerate(classes):

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    print(
        f"Processing {class_name}..."
    )

    files = os.listdir(
        folder_path
    )

    count = 0

    for filename in files:

        file_path = os.path.join(
            folder_path,
            filename
        )

        image = cv2.imread(
            file_path
        )

        if image is None:
            continue


        landmarks = extract_landmarks(
            image
        )


        if landmarks is None:
            continue


        # Must always be 126
        if len(landmarks) != 126:
            continue


        X.append(
            landmarks
        )

        y.append(
            class_index
        )

        count += 1


    print(
        f"  Valid images: {count}"
    )


# =========================================================
# CONVERT TO NUMPY
# =========================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)


print()
print(
    "Dataset shape:",
    X.shape
)


if len(X) == 0:

    print(
        "ERROR: No hand landmarks found."
    )

    print(
        "Check your dataset images."
    )

    exit()


# =========================================================
# CHECK INPUT SIZE
# =========================================================

if X.shape[1] != 126:

    print(
        "ERROR: Expected 126 features."
    )

    print(
        "Actual:",
        X.shape[1]
    )

    exit()


# =========================================================
# ONE-HOT ENCODING
# =========================================================

y = to_categorical(
    y,
    num_classes=len(classes)
)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print()
print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# =========================================================
# MODEL
# =========================================================

model = Sequential([

    Input(
        shape=(126,)
    ),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        len(classes),
        activation="softmax"
    )

])


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


model.summary()


# =========================================================
# TRAIN
# =========================================================

print()
print("Training started...")
print()


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=30,

    batch_size=32,

    verbose=1

)


# =========================================================
# TEST
# =========================================================

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


print()

print(
    "Test Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


model.save(
    MODEL_PATH
)


print()
print(
    "Model saved:"
)

print(
    MODEL_PATH
)


# =========================================================
# SAVE CLASSES
# =========================================================

with open(
    CLASSES_PATH,
    "w"
) as file:

    for class_name in classes:

        file.write(
            class_name + "\n"
        )


print(
    "Classes saved:"
)

print(
    CLASSES_PATH
)


print()
print(
    "Training completed successfully!"
)