import cv2
import numpy as np
import pickle
import os
import warnings
from hand_recognizer import HandRecognizer

# Suprimir advertencias específicas de scikit-learn sobre feature names
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")

class SignLanguageRecognizer:
    def __init__(self, model_path='gesture_model.pkl'):
        # Cargar el modelo entrenado
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Inicializar el reconocedor de manos
        self.hand_recognizer = HandRecognizer(max_num_hands=1)
        
        # Variables para la formación de palabras y oraciones
        self.current_letter = ''
        self.written_word = ''
        self.sentence = ''
        
        # Contador para estabilización
        self.last_predictions = []
        self.max_predictions = 5  # Cantidad de predicciones a considerar
        
        print("[INFO] Modelo cargado correctamente")
        print("[INFO] Controles:")
        print("       's' - Agregar letra actual a la palabra")
        print("       'd' - Borrar última letra")
        print("       'espacio' - Agregar palabra actual a la oración")
        print("       'r' - Reiniciar oración")
        print("       'q' - Salir")

    def normalize_landmarks(self, landmarks):
        """Normaliza los landmarks igual que en data_collector"""
        base_x, base_y = landmarks[0]
        norm_landmarks = [(x - base_x, y - base_y) for x, y in landmarks]
        max_val = max([abs(x) for x, y in norm_landmarks] + [abs(y) for x, y in norm_landmarks])
        if max_val > 0:
            norm_landmarks = [(x / max_val, y / max_val) for x, y in norm_landmarks]
        # Aplanar la lista para el modelo
        flat_landmarks = [coord for point in norm_landmarks for coord in point]
        return flat_landmarks

    def get_stable_prediction(self):
        """Devuelve la predicción más común de las últimas N predicciones"""
        if not self.last_predictions:
            return ""
            
        # Contar ocurrencias
        from collections import Counter
        counter = Counter(self.last_predictions)
        
        # Devolver la más común
        return counter.most_common(1)[0][0]

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Reflejar para vista tipo espejo
            frame = cv2.flip(frame, 1)

            # Detectar manos y obtener landmarks
            frame, hands_data = self.hand_recognizer.detect_hands(frame)

            # Procesar si hay manos detectadas
            if hands_data:
                # Normalizar landmarks para el modelo
                landmarks = hands_data[0]['landmarks']
                norm_landmarks = self.normalize_landmarks(landmarks)
                
                # Convertir a pandas DataFrame con los nombres de características correctos si es necesario
                norm_landmarks_array = np.array([norm_landmarks])
                
                # Predecir letra (con manejo de advertencias)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prediction = self.model.predict(norm_landmarks_array)[0]
                
                # Actualizar lista de predicciones recientes
                self.last_predictions.append(prediction)
                if len(self.last_predictions) > self.max_predictions:
                    self.last_predictions.pop(0)
                
                # Obtener predicción estable
                self.current_letter = self.get_stable_prediction()
                
                # Extraer landmarks de la estructura hands_data
                landmarks = hands_data[0]['landmarks']
                
                # El rectángulo ya se dibuja en HandRecognizer, solo necesitamos sus coordenadas
                # para colocar el texto encima
                height, width, _ = frame.shape
                cx_list = [int(lm[0] * width) for lm in landmarks]
                cy_list = [int(lm[1] * height) for lm in landmarks]
                x_min, x_max = min(cx_list), max(cx_list)
                y_min, y_max = min(cy_list), max(cy_list)

                # Calcular tamaño del texto para centrarlo
                (text_width, _), _ = cv2.getTextSize(self.current_letter, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                text_x = x_min + (x_max - x_min - text_width) // 2
                text_y = y_min - 20  # 20 píxeles arriba del rectángulo

                # Mostrar letra actual centrada encima del rectángulo
                cv2.putText(frame, self.current_letter, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)


            # Información en pantalla
            cv2.putText(frame, f"Palabra: {self.written_word}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Oracion: {self.sentence}", (10, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Reconocimiento de Lenguaje de Senias", frame)

            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and self.current_letter:
                self.written_word += self.current_letter
                print(f"[+] Letra '{self.current_letter}' añadida")
            
            elif key == ord('d') and self.written_word:
                removed = self.written_word[-1]
                self.written_word = self.written_word[:-1]
                print(f"[-] Letra '{removed}' eliminada")
            
            elif key == ord(' '):
                if self.written_word:
                    self.sentence += self.written_word + " "
                    print(f"[+] Palabra '{self.written_word}' añadida a la oración")
                    self.written_word = ""
            
            elif key == ord('r'):
                self.sentence = ""
                print("[!] Oración reiniciada")
            
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = 'data/gesture_model.pkl'  # Cambia esto si tu modelo está en otra ubicación
    
    try:
        recognizer = SignLanguageRecognizer(model_path)
        recognizer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrate de haber entrenado el modelo primero ejecutando train_model.ipynb")
    except Exception as e:
        print(f"Error inesperado: {e}")