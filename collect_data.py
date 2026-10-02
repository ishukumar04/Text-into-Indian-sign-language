import cv2
import os
import time

# ============================================================
# ISL DATASET COLLECTION
# Automatically captures 300 images for each sign
# ============================================================

# Signs/classes to collect
SIGNS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "HELP",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F"
]

# Number of images per sign
IMAGES_PER_SIGN = 300

# Time between captures (seconds)
CAPTURE_DELAY = 0.08

# Dataset folder
DATASET_DIR = "dataset"


def create_folders():
    """Create folders for all signs."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    for sign in SIGNS:
        os.makedirs(os.path.join(DATASET_DIR, sign), exist_ok=True)


def get_next_number(folder):
    """Find the next available image number."""
    files = os.listdir(folder)

    numbers = []

    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file)[0]

            if name.isdigit():
                numbers.append(int(name))

    if numbers:
        return max(numbers) + 1

    return 1


def collect_sign(sign, camera):
    """Capture images for one sign."""

    folder = os.path.join(DATASET_DIR, sign)
    os.makedirs(folder, exist_ok=True)

    start_number = get_next_number(folder)

    # How many images already exist
    existing_images = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    remaining = IMAGES_PER_SIGN - existing_images

    if remaining <= 0:
        print(f"\n{sign}: Already has {IMAGES_PER_SIGN} images.")
        return

    print("\n" + "=" * 50)
    print(f"SIGN: {sign}")
    print(f"Images already collected: {existing_images}")
    print(f"Images remaining: {remaining}")
    print("=" * 50)

    print("\nGet ready...")
    time.sleep(2)

    print("Starting in:")
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)

    print("\nCAPTURING! ✋")

    count = existing_images

    for i in range(remaining):

        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read camera.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        image_number = start_number + i

        filename = os.path.join(
            folder,
            f"{image_number:03d}.jpg"
        )

        # Save image
        cv2.imwrite(filename, frame)

        count += 1

        # Display information
        display = frame.copy()

        cv2.putText(
            display,
            f"Sign: {sign}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Image: {count}/{IMAGES_PER_SIGN}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "Keep your hand visible",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Press Q to stop",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow("ISL Dataset Collection", display)

        # Wait
        key = cv2.waitKey(int(CAPTURE_DELAY * 1000)) & 0xFF

        # Press Q to stop
        if key == ord("q"):
            print("\nCollection stopped by user.")
            break

    print(f"\nFinished {sign}: {count} images collected.")


def main():

    print("=" * 60)
    print("       INDIAN SIGN LANGUAGE DATASET COLLECTION")
    print("=" * 60)

    print("\nSigns:")
    for i, sign in enumerate(SIGNS, 1):
        print(f"{i}. {sign}")

    print(f"\nImages per sign: {IMAGES_PER_SIGN}")
    print(f"Total images: {len(SIGNS) * IMAGES_PER_SIGN}")

    # Create folders
    create_folders()

    # Open webcam
    print("\nOpening webcam...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("\nERROR: Cannot open webcam.")
        print("Check that your camera is connected and not being used by another app.")
        return

    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nCamera opened successfully.")

    print("\nPress ENTER to start collecting data.")
    input()

    try:

        for sign in SIGNS:

            collect_sign(sign, camera)

            print("\nNext sign will start soon...")
            time.sleep(2)

    except KeyboardInterrupt:

        print("\n\nCollection interrupted.")

    finally:

        camera.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("DATASET COLLECTION COMPLETE!")
    print("=" * 60)

    print("\nYour dataset is stored here:")
    print(os.path.abspath(DATASET_DIR))

    print("\nNext step:")
    print("Run: python train.py")


if __name__ == "__main__":
    main()