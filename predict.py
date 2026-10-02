import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "model/isl_model.h5"
CLASSES_PATH = "model/classes.txt"


# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# LOAD CLASSES
# =========================================================

with open(CLASSES_PATH, "r") as file:
    classes = [line.strip() for line in file if line.strip()]

print("Classes:")
for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# FUNCTION: EXTRACT HAND LANDMARKS
# =========================================================

def get_landmarks(hand_landmarks):

    landmarks = []

    for point in hand_landmarks.landmark:
        landmarks.append(point.x)
        landmarks.append(point.y)
        landmarks.append(point.z)

    return landmarks


# =========================================================
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print()
print("Webcam started.")
print("Two-hand recognition enabled.")
print("Press Q to quit.")
print()


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe
    result = hands.process(rgb)


    prediction_text = "No hand detected"


    # =====================================================
    # HAND DETECTION
    # =====================================================

    if result.multi_hand_landmarks:

        # Store left and right hand separately
        left_hand = None
        right_hand = None


        # =================================================
        # GET EACH HAND
        # =================================================

        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # MediaPipe handedness
            hand_label = handedness.classification[0].label


            # Get 63 values
            landmarks = get_landmarks(
                hand_landmarks
            )


            if hand_label == "Left":
                left_hand = landmarks

            elif hand_label == "Right":
                right_hand = landmarks


        # =================================================
        # CREATE TWO-HAND INPUT
        # =================================================

        # If left hand is missing → 63 zeros
        if left_hand is None:
            left_hand = [0.0] * 63


        # If right hand is missing → 63 zeros
        if right_hand is None:
            right_hand = [0.0] * 63


        # Left + Right = 126 values
        landmarks = left_hand + right_hand


        # =================================================
        # NUMPY
        # =================================================

        input_data = np.array(
            landmarks,
            dtype=np.float32
        )


        # Add batch dimension
        input_data = np.expand_dims(
            input_data,
            axis=0
        )


        # =================================================
        # PREDICTION
        # =================================================

        predictions = model.predict(
            input_data,
            verbose=0
        )


        predicted_index = np.argmax(
            predictions[0]
        )


        confidence = predictions[0][
            predicted_index
        ]


        # Safety check
        if predicted_index < len(classes):

            predicted_class = classes[
                predicted_index
            ]

        else:

            predicted_class = "Unknown"


        prediction_text = (
            f"{predicted_class} "
            f"({confidence * 100:.1f}%)"
        )


    # =====================================================
    # DISPLAY PREDICTION
    # =====================================================

    cv2.putText(
        frame,
        prediction_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # =====================================================
    # DISPLAY HAND COUNT
    # =====================================================

    if result.multi_hand_landmarks:

        hand_count = len(
            result.multi_hand_landmarks
        )

        hand_text = f"Hands detected: {hand_count}"

    else:

        hand_text = "Hands detected: 0"


    cv2.putText(
        frame,
        hand_text,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "ISL Sign Language Recognition",
        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Webcam stopped.")