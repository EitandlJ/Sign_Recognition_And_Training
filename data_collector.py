import cv2
import csv
import os
from hand_recognizer import HandRecognizer  

class DataCollector:
    def __init__(self, output_file='data/dataset.csv', labels_file='data/labels.csv'):
        self.output_file = output_file
        self.labels_file = labels_file
        self.hand_recognizer = HandRecognizer()
        self.cap = cv2.VideoCapture(0)

        # Crear directorio de datos si no existe
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Crear archivo de datos si no existe
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
                writer.writerow(header)

        # Crear archivo de etiquetas si no existe
        if not os.path.exists(self.labels_file):
            os.makedirs(os.path.dirname(self.labels_file), exist_ok=True)
            with open(self.labels_file, 'w', newline='') as f:
                pass

    def normalize_landmarks(self, landmarks):
        base_x, base_y = landmarks[0]
        norm_landmarks = [(x - base_x, y - base_y) for x, y in landmarks]
        max_val = max([abs(x) for x, y in norm_landmarks] + [abs(y) for x, y in norm_landmarks])
        if max_val > 0:
            norm_landmarks = [(x / max_val, y / max_val) for x, y in norm_landmarks]
        return norm_landmarks

    def save_label(self, label):
        with open(self.labels_file, 'r') as f:
            labels = [line.strip() for line in f]
        if label not in labels:
            with open(self.labels_file, 'a') as f:
                f.write(label + '\n')
            print(f"[INFO] Nueva etiqueta registrada: {label}")

    def collect(self, label):
        print(f"[INFO] Recolectando datos para: {label} (presiona 's' para guardar, 'q' para salir)")
        self.save_label(label)
        
        # Contador para esta sesión
        sample_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, hands_data = self.hand_recognizer.detect_hands(frame)

            if hands_data:
                landmarks = hands_data[0]['landmarks']
                norm_coords = self.normalize_landmarks(landmarks)

            # Mostrar etiqueta y contador
            cv2.putText(frame, f"Etiqueta: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Muestras: {sample_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Recolección", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and hands_data:
                sample_count += 1
                with open(self.output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [v for point in norm_coords for v in point] + [label]
                    writer.writerow(row)
                print(f"[+] Gesto guardado para '{label}': Muestra #{sample_count}")
            elif key == ord('q'):
                break

        print(f"[INFO] Sesión finalizada. Se guardaron {sample_count} muestras para '{label}'")
        self.cap.release()
        cv2.destroyAllWindows()