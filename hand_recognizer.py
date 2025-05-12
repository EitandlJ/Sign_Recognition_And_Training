import cv2
import mediapipe as mp

class HandRecognizer:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def detect_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        hands_data = []

        height, width, _ = frame.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Convertir puntos normalizados a pixeles
                cx_list = [int(lm.x * width) for lm in hand_landmarks.landmark]
                cy_list = [int(lm.y * height) for lm in hand_landmarks.landmark]
                x_min, x_max = min(cx_list), max(cx_list)
                y_min, y_max = min(cy_list), max(cy_list)

                # Dibujar marco rectangular en blanco
                cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 255, 255), 2)

                # Dibujar landmarks y conexiones en blanco
                for i in range(len(hand_landmarks.landmark)):
                    cx, cy = cx_list[i], cy_list[i]
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    x0, y0 = cx_list[start_idx], cy_list[start_idx]
                    x1, y1 = cx_list[end_idx], cy_list[end_idx]
                    cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)

                # Extraer datos
                label = hand_label.classification[0].label  # 'Left' o 'Right'
                hand_data = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                hands_data.append({
                    'label': label,
                    'landmarks': hand_data
                })

        return frame, hands_data


# Función para obtener los puntos de los landmarks de la mano
def get_hand_landmarks(frame):
    recognizer = HandRecognizer(max_num_hands=1)
    _, hands = recognizer.detect_hands(frame)
    if hands:
        landmarks = hands[0]['landmarks']  # [(x1, y1), ..., (x21, y21)]
        flat_landmarks = [coord for point in landmarks for coord in point]  # [x1, y1, x2, y2, ..., x21, y21]
        return flat_landmarks
    return None
