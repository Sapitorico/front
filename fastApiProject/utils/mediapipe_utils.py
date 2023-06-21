#!/usr/bin/python3
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


class MediaPipeUtils:
    def __init__(self):
        self.offset=10

    @staticmethod
    def Hands_model_configuration(image_mode, max_hand, complexity):
        hands = mp_hands.Hands(
            static_image_mode=image_mode,
            max_num_hands=max_hand,
            model_complexity=complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return hands

    @staticmethod
    def Hands_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = model.process(image)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, result

    @staticmethod
    def Detect_hand_type(hand_type, results, positions, copie_img):
        land_mark = np.zeros(21 * 3)
        for hand_index, hand_info in enumerate(results.multi_handedness):
            hand_types = hand_info.classification[0].label
            if hand_types == hand_type:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        alto, ancho, c = copie_img.shape
                        positions.append((lm.x * ancho, lm.y * alto, lm.z * ancho))
                        land_mark = np.array([lm.x, lm.y, lm.z]).flatten()
        return positions, land_mark

    def Draw_Bound_Boxes(self, positions, frame):
        x_min = int(min(positions, key=lambda x: x[0])[0])
        y_min = int(min(positions, key=lambda x: x[1])[1])
        x_max = int(max(positions, key=lambda x: x[0])[0])
        y_max = int(max(positions, key=lambda x: x[1])[1])
        width = x_max - x_min
        height = y_max - y_min
        x1, y1 = x_min, y_min
        x2, y2 = x_min + width, y_min + height
        if y1 - self.offset - 15 >= 0 and y2 - self.offset + 40 <= frame.shape[
            0] and x1 - self.offset - 40 >= 0 and x2 - self.offset + 50 <= \
                frame.shape[1]:
            cv2.rectangle(frame, (x1 - self.offset - 40, y1 - self.offset - 15), (x2 - self.offset + 50, y2 - self.offset + 40),
                          (0, 255, 0), 3)