# Sign Language Recognition

Este proyecto es una aplicación para el reconocimiento de gestos con las manos. Permite entrenar el modelo para reconocer nuevos gestos, realizar el entrenamiento con los datos recolectados y ejecutar una aplicación en tiempo real para predecir gestos.

## Requisitos
Asegúrate de tener instaladas las siguientes dependencias. Puedes instalarlas ejecutando:

pip install -r [requirements.txt]

## Dependencias

opencv-python: Para procesamiento de imágenes y video.
numpy: Para operaciones numéricas.
mediapipe: Para detección de manos y extracción de landmarks.
scikit-learn: Para el entrenamiento y predicción del modelo.
pandas: Para manipulación de datos.
seaborn: Para visualización de datos.
jupyter: Para ejecutar notebooks de Jupyter.

## Uso del código

1. **Recolectar nuevos gestos**:
   - Utiliza el archivo `main.py` para recolectar datos de gestos personalizados y almacenarlos en un archivo CSV (`dataset.csv`) y almacenar las etiquetas de los mismos en (`labels.csv`), ambos dentro de la carpeta (`/data`).

   

2. **Entrenar el modelo**:

   - Usa un notebook de Jupyter para entrenar un modelo con los datos recolectados.

3. **Predicción en tiempo real**:

   - Ejecuta el archivo `live_prediction.py` para predecir gestos en tiempo real utilizando una cámara web.

