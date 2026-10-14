import cv2
import os
import time
import numpy as np
from urllib.request import urlretrieve

import mediapipe as mp
import screen_brightness_control as sbc

from datetime import datetime

from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision.core.image import Image as MpImage
from mediapipe.tasks.python.vision.core.image import ImageFormat

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_landmarker_model():

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "hand_landmarker.task"
    )

    if not os.path.exists(model_path):
        print("Downloading hand tracking model...")
        urlretrieve(HAND_MODEL_URL, model_path)

    return model_path


# ==========================================
# SMARTGLOW CONFIGURATION
# ==========================================

MIN_BRIGHTNESS = 15
MAX_BRIGHTNESS = 100

BRIGHTNESS_STEP = 5

AUTO_UPDATE_INTERVAL = 2.0
GESTURE_COOLDOWN = 0.8

SMOOTHING_FACTOR = 0.15

# Prevent frequent changes caused by small variations.
MIN_CHANGE_THRESHOLD = 2

CAMERA_INDEX = 0


# ==========================================
# BRIGHTNESS CONTROLLER
# ==========================================

class SmartBrightness:

    def __init__(self):

        self.current_brightness = None
        self.target_brightness = None

        self.auto_mode = True
        self.last_auto_update = 0

        self.last_gesture_time = 0

        self.status = "Starting SmartGlow"

        self.initialize_brightness()

    def initialize_brightness(self):

        try:
            brightness = sbc.get_brightness()

            if isinstance(brightness, list):
                brightness = brightness[0]

            self.current_brightness = int(brightness)
            self.target_brightness = int(brightness)

            print(
                f"Current brightness: "
                f"{self.current_brightness}%"
            )

        except Exception as error:

            print("Could not read brightness:", error)

            self.current_brightness = 50
            self.target_brightness = 50

    def clamp(self, value):

        return max(
            MIN_BRIGHTNESS,
            min(MAX_BRIGHTNESS, int(value))
        )

    def set_target(self, value):

        self.target_brightness = self.clamp(value)

    def change_brightness(self, amount):

        self.set_target(
            self.target_brightness + amount
        )

        self.status = (
            f"Manual target: {self.target_brightness}%"
        )

    def toggle_auto(self, enabled):

        self.auto_mode = enabled

        if enabled:
            self.status = "Automatic mode resumed"
        else:
            self.status = "Automatic mode paused"

    def update_automatic_brightness(self, light_level):

        now = time.time()

        if not self.auto_mode:
            return

        if now - self.last_auto_update < AUTO_UPDATE_INTERVAL:
            return

        self.last_auto_update = now

        hour = datetime.now().hour

        # Estimate an appropriate brightness from the
        # webcam's grayscale image brightness.
        #
        # The camera image is only an indirect estimate
        # of ambient light.

        light_target = MIN_BRIGHTNESS + (
            light_level / 255
        ) * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)

        # Time-aware brightness preferences.
        if 0 <= hour < 6:
            time_target = 25

        elif 6 <= hour < 10:
            time_target = 55

        elif 10 <= hour < 17:
            time_target = 80

        elif 17 <= hour < 21:
            time_target = 55

        else:
            time_target = 30

        # Combine the light estimate and time preference.
        combined_target = (
            0.70 * light_target
            + 0.30 * time_target
        )

        self.set_target(combined_target)

        self.status = (
            f"Automatic target: "
            f"{self.target_brightness}%"
        )

    def smooth_brightness(self):

        if self.current_brightness is None:
            return

        difference = (
            self.target_brightness
            - self.current_brightness
        )

        if abs(difference) < MIN_CHANGE_THRESHOLD:
            return

        step = difference * SMOOTHING_FACTOR

        # Ensure the value changes by at least 1.
        if abs(step) < 1:
            step = 1 if difference > 0 else -1

        next_brightness = round(
            self.current_brightness + step
        )

        next_brightness = self.clamp(next_brightness)

        if next_brightness == self.current_brightness:
            return

        try:

            sbc.set_brightness(next_brightness)

            self.current_brightness = next_brightness

        except Exception as error:

            self.status = "Display control error"

            print("Brightness update failed:", error)


# ==========================================
# HAND GESTURE RECOGNITION
# ==========================================

class HandGestureController:

    def __init__(self):

        self.model_path = ensure_hand_landmarker_model()

        self.mp_hands = None
        self.mp_draw = None
        self.hands = None
        self.landmarker = None

        self.previous_gesture = None
        self.stable_frames = 0

        self.initialize_hand_tracker()

    def initialize_hand_tracker(self):

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.60
            )
            return

        options = HandLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        self.mp_draw = mp_drawing

    def count_fingers(self, landmarks):

        fingers = []

        # Index, middle, ring and little fingers.
        finger_pairs = [
            (8, 6),
            (12, 10),
            (16, 14),
            (20, 18)
        ]

        for tip, joint in finger_pairs:

            if landmarks[tip].y < landmarks[joint].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Thumb is approximately horizontal.
        # This simple check assumes the hand is visible
        # with its palm generally facing the camera.
        thumb_tip = landmarks[4]
        thumb_joint = landmarks[3]

        if thumb_tip.x < thumb_joint.x:
            fingers.append(1)
        else:
            fingers.append(0)

        return fingers

    def identify_gesture(self, fingers):

        total = sum(fingers)

        if total == 0:
            return "FIST"

        if total == 5:
            return "OPEN_PALM"

        if fingers == [1, 0, 0, 0, 0]:
            return "ONE_FINGER"

        if fingers == [1, 1, 0, 0, 0]:
            return "TWO_FINGERS"

        return "UNKNOWN"

    def process_frame(self, frame):

        if self.landmarker is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = MpImage(
                image_format=ImageFormat.SRGB,
                data=frame_rgb
            )

            result = self.landmarker.detect(image)

            gesture = "NO HAND"

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS
                )

                fingers = self.count_fingers(hand_landmarks)
                gesture = self.identify_gesture(fingers)

            return frame, gesture

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(frame_rgb)

        gesture = "NO HAND"

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            fingers = self.count_fingers(landmarks)

            gesture = self.identify_gesture(fingers)

        return frame, gesture

    def close(self):

        if self.hands is not None:
            self.hands.close()

        if self.landmarker is not None:
            self.landmarker.close()


# ==========================================
# LIGHT ESTIMATION
# ==========================================

def estimate_light(frame):

    # Use the center of the image to reduce the influence
    # of bright areas near the edges.

    height, width = frame.shape[:2]

    x1 = width // 4
    x2 = 3 * width // 4

    y1 = height // 4
    y2 = 3 * height // 4

    center = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(
        center,
        cv2.COLOR_BGR2GRAY
    )

    # Percentile-based estimate is less sensitive
    # to a few extremely bright pixels.

    light_level = np.percentile(gray, 50)

    return float(light_level)


# ==========================================
# ON-SCREEN INFORMATION
# ==========================================

def draw_text(frame, text, position, scale=0.65):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    brightness = SmartBrightness()

    gesture_controller = HandGestureController()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")

        gesture_controller.close()

        return

    print("\n" + "=" * 45)
    print("          SMARTGLOW CONTROLLER")
    print("=" * 45)

    print("ONE FINGER  : Increase brightness")
    print("TWO FINGERS : Decrease brightness")
    print("CLOSED FIST : Pause automatic mode")
    print("OPEN PALM   : Resume automatic mode")
    print("A           : Toggle automatic mode")
    print("Q           : Quit")
    print("=" * 45 + "\n")

    # Smooth the light estimate to reduce sudden changes.
    smoothed_light = None

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print("Could not read webcam frame.")
                break

            # Mirror the image for a natural webcam preview.
            frame = cv2.flip(frame, 1)

            # Estimate lighting.
            raw_light = estimate_light(frame)

            if smoothed_light is None:
                smoothed_light = raw_light
            else:
                smoothed_light = (
                    0.15 * raw_light
                    + 0.85 * smoothed_light
                )

            # Update automatic target brightness.
            brightness.update_automatic_brightness(
                smoothed_light
            )

            # Process hand gestures.
            frame, gesture = (
                gesture_controller.process_frame(frame)
            )

            now = time.time()

            # Require a gesture to remain stable for
            # several frames before acting.
            if gesture == gesture_controller.previous_gesture:
                gesture_controller.stable_frames += 1
            else:
                gesture_controller.previous_gesture = gesture
                gesture_controller.stable_frames = 1

            if (
                gesture_controller.stable_frames >= 4
                and now - brightness.last_gesture_time
                >= GESTURE_COOLDOWN
            ):

                if gesture == "ONE_FINGER":

                    brightness.change_brightness(
                        BRIGHTNESS_STEP
                    )

                    brightness.last_gesture_time = now

                elif gesture == "TWO_FINGERS":

                    brightness.change_brightness(
                        -BRIGHTNESS_STEP
                    )

                    brightness.last_gesture_time = now

                elif gesture == "FIST":

                    brightness.toggle_auto(False)

                    brightness.last_gesture_time = now

                elif gesture == "OPEN_PALM":

                    brightness.toggle_auto(True)

                    brightness.last_gesture_time = now

            # Smoothly move towards the target.
            brightness.smooth_brightness()

            # Show information.
            draw_text(
                frame,
                f"Brightness: {brightness.current_brightness}%",
                (20, 35)
            )

            draw_text(
                frame,
                f"Target: {brightness.target_brightness}%",
                (20, 65)
            )

            draw_text(
                frame,
                f"Mode: {'AUTO' if brightness.auto_mode else 'MANUAL PAUSE'}",
                (20, 95)
            )

            draw_text(
                frame,
                f"Gesture: {gesture}",
                (20, 125)
            )

            draw_text(
                frame,
                f"Estimated light: {smoothed_light:.0f}/255",
                (20, 155)
            )

            draw_text(
                frame,
                brightness.status[:55],
                (20, 185),
                0.5
            )

            cv2.imshow(
                "SmartGlow - Smart Brightness Controller",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("a"):

                brightness.toggle_auto(
                    not brightness.auto_mode
                )

            elif key == 82 or key == ord("+"):

                brightness.change_brightness(
                    BRIGHTNESS_STEP
                )

            elif key == 84 or key == ord("-"):

                brightness.change_brightness(
                    -BRIGHTNESS_STEP
                )

    except KeyboardInterrupt:

        print("\nSmartGlow stopped.")

    finally:

        camera.release()

        gesture_controller.close()

        cv2.destroyAllWindows()

        print("Webcam released.")
        print("SmartGlow closed.")


if __name__ == "__main__":

    main()

