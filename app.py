from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "model/isl_model.h5"
CLASSES_PATH = "model/classes.txt"


# =========================================================
# LOAD MODEL
# =========================================================

model = None
classes = []


if os.path.exists(MODEL_PATH):

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")
    print("Model input shape:", model.input_shape)

else:

    print("WARNING: Model not found.")
    print("Run train.py first.")


# =========================================================
# LOAD CLASSES
# =========================================================

if os.path.exists(CLASSES_PATH):

    with open(
        CLASSES_PATH,
        "r"
    ) as file:

        classes = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    print("\nClasses:")

    for index, class_name in enumerate(classes):

        print(
            f"{index}: {class_name}"
        )


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
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# STATUS
# =========================================================

@app.route("/status")
def status():

    return jsonify({

        "server": "running",

        "model_loaded": model is not None,

        "model_input": (
            str(model.input_shape)
            if model is not None
            else None
        ),

        "classes": classes

    })


# =========================================================
# EXTRACT TWO-HAND LANDMARKS
# =========================================================

def extract_landmarks(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    result = hands.process(
        image_rgb
    )


    # -----------------------------------------------------
    # Create empty 63-value arrays
    # -----------------------------------------------------

    left_hand = np.zeros(
        63,
        dtype=np.float32
    )

    right_hand = np.zeros(
        63,
        dtype=np.float32
    )


    # -----------------------------------------------------
    # No hand detected
    # -----------------------------------------------------

    if not result.multi_hand_landmarks:

        return None


    # -----------------------------------------------------
    # Extract detected hands
    # -----------------------------------------------------

    for hand_landmarks, handedness in zip(

        result.multi_hand_landmarks,

        result.multi_handedness

    ):

        landmarks = []


        for point in hand_landmarks.landmark:

            landmarks.extend([

                point.x,

                point.y,

                point.z

            ])


        landmarks = np.array(

            landmarks,

            dtype=np.float32

        )


        # -------------------------------------------------
        # Identify Left / Right
        # -------------------------------------------------

        hand_type = (

            handedness
            .classification[0]
            .label

        )


        if hand_type == "Left":

            left_hand = landmarks


        elif hand_type == "Right":

            right_hand = landmarks


    # -----------------------------------------------------
    # Combine:
    #
    # Left  = 63
    # Right = 63
    #
    # Total = 126
    # -----------------------------------------------------

    combined = np.concatenate([

        left_hand,

        right_hand

    ])


    return combined


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------------------
    # Model check
    # -----------------------------------------------------

    if model is None:

        return jsonify({

            "success": False,

            "message":
            "Model not found. Run train.py first."

        })


    # -----------------------------------------------------
    # Classes check
    # -----------------------------------------------------

    if len(classes) == 0:

        return jsonify({

            "success": False,

            "message":
            "Classes not found."

        })


    # -----------------------------------------------------
    # Image check
    # -----------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "message":
            "No image received."

        })


    file = request.files["image"]


    image_bytes = file.read()


    image_array = np.frombuffer(

        image_bytes,

        dtype=np.uint8

    )


    image = cv2.imdecode(

        image_array,

        cv2.IMREAD_COLOR

    )


    if image is None:

        return jsonify({

            "success": False,

            "message":
            "Invalid image."

        })


    # -----------------------------------------------------
    # Extract 126 features
    # -----------------------------------------------------

    landmarks = extract_landmarks(
        image
    )


    # -----------------------------------------------------
    # No hand
    # -----------------------------------------------------

    if landmarks is None:

        return jsonify({

            "success": True,

            "detected": False,

            "index": -1,

            "prediction": "NO HAND",

            "confidence": 0

        })


    # -----------------------------------------------------
    # IMPORTANT:
    # Model expects 126
    # -----------------------------------------------------

    if len(landmarks) != 126:

        return jsonify({

            "success": False,

            "message":
            f"Expected 126 features, "
            f"got {len(landmarks)}."

        })


    # -----------------------------------------------------
    # Reshape
    # -----------------------------------------------------

    landmarks = landmarks.reshape(
        1,
        126
    )


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = model.predict(

        landmarks,

        verbose=0

    )[0]


    # -----------------------------------------------------
    # Get index
    # -----------------------------------------------------

    predicted_index = int(

        np.argmax(
            predictions
        )

    )


    # -----------------------------------------------------
    # Check index
    # -----------------------------------------------------

    if predicted_index >= len(classes):

        return jsonify({

            "success": False,

            "message":
            "Invalid class index."

        })


    # -----------------------------------------------------
    # Class
    # -----------------------------------------------------

    predicted_class = classes[
        predicted_index
    ]


    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = float(

        predictions[
            predicted_index
        ]

    )


    confidence_percent = (

        confidence * 100

    )


    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    if confidence < 0.60:

        display_class = "UNKNOWN"

    else:

        display_class = predicted_class


    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return jsonify({

        "success": True,

        "detected": True,

        "index": predicted_index,

        "prediction": display_class,

        "class": predicted_class,

        "confidence": round(

            confidence_percent,

            2

        )

    })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print("==============================")
    print(" ISL Recognition Website")
    print("==============================")
    print()
    print("Model:", MODEL_PATH)
    print("Classes:", CLASSES_PATH)
    print()

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )