"""
Sistema de Traducción en Tiempo Real de Lengua de Señas (Abecedario LSM)
Versión mejorada con:
- Interfaz Tkinter
- Voz (pyttsx3)
- Reporte de precisión en pantalla
- Bitácora de predicciones
"""

import cv2
import csv
import os
import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pyttsx3
from datetime import datetime

# ================================
# CONFIGURACIÓN GLOBAL
# ================================
DATASET = "abecedario.csv"
MODELO = "modelo_abecedario.joblib"
LOGFILE = "bitacora_predicciones.csv"

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Motor de voz
voz = pyttsx3.init()
voz.setProperty("rate", 150)

# ================================
# FUNCIONES BASE
# ================================
def extraer_landmarks(results):
    """Extrae 63 valores (21 puntos * XYZ) de la mano detectada."""
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0]
    coords = []

    for p in lm.landmark:
        coords.append(p.x)
        coords.append(p.y)
        coords.append(p.z)

    return coords


# ================================
# RECOLECCIÓN DE DATOS
# ================================
def recolectar(letra, muestras=300):
    """Recolecta datos para una letra específica A–Z."""
    if not os.path.exists(DATASET):
        with open(DATASET, "w", newline="") as f:
            writer = csv.writer(f)
            columnas = ["label"] + [f"v{i}" for i in range(21*3)]
            writer.writerow(columnas)

    cap = cv2.VideoCapture(0)
    manos = mp_hands.Hands()

    total = 0
    while total < muestras:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = manos.process(rgb)

        feat = extraer_landmarks(results)
        if feat is not None:
            with open(DATASET, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([letra] + feat)
            total += 1

        cv2.putText(frame, f"Recolectando: {letra} ({total}/{muestras})",
                    (10, 40), 1, 1.5, (0,255,0), 2)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Recolección", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Completado", f"Recolección de {letra} finalizada.")


# ================================
# ENTRENAMIENTO
# ================================
def entrenar_modelo(label_precision):
    """Entrena un modelo Random Forest y muestra precisión en pantalla."""
    if not os.path.exists(DATASET):
        messagebox.showerror("Error", "No existe dataset. Recolecta datos primero.")
        return

    df = pd.read_csv(DATASET)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=300)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, pred)

    joblib.dump(modelo, MODELO)

    label_precision.config(text=f"Precisión del modelo: {precision*100:.2f}%")
    messagebox.showinfo("Éxito", "Modelo entrenado y guardado exitosamente.")


# ================================
# REGISTRO EN BITÁCORA
# ================================
def registrar_prediccion(letra):
    """Guarda predicción en un archivo CSV con fecha y hora."""
    existe = os.path.exists(LOGFILE)

    with open(LOGFILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["Fecha", "Letra"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), letra])


# ================================
# RECONOCIMIENTO EN TIEMPO REAL
# ================================
def reconocer():
    """Reconoce letras A–Z en tiempo real y usa voz."""
    if not os.path.exists(MODELO):
        messagebox.showerror("Error", "Primero entrena el modelo.")
        return

    modelo = joblib.load(MODELO)
    cap = cv2.VideoCapture(0)
    manos = mp_hands.Hands()

    last_letter = ""

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = manos.process(rgb)

        feat = extraer_landmarks(results)
        letra = ""

        if feat is not None:
            X = np.array(feat).reshape(1, -1)
            letra = modelo.predict(X)[0]

            # Cuando cambia la letra, se dice en voz y se registra
            if letra != last_letter:
                voz.say(letra)
                voz.runAndWait()
                registrar_prediccion(letra)
                last_letter = letra

        cv2.putText(frame, f"Letra: {letra}", (10, 40),
                    1, 2, (255,255,255), 2)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Reconocimiento LSM", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ================================
# INTERFAZ TKINTER
# ================================
def interfaz():
    ventana = tk.Tk()
    ventana.title("Reconocimiento de Abecedario LSM")
    ventana.geometry("450x300")

    tk.Label(ventana, text="Sistema de Traducción en Tiempo Real - LSM",
             font=("Arial", 14)).pack(pady=10)

    # Botón recolectar
    def pedir_letra():
        letra = entrada_letra.get().upper()
        if len(letra) != 1 or not letra.isalpha():
            messagebox.showerror("Error", "Ingresa solo una letra A-Z.")
            return
        recolectar(letra)

    tk.Label(ventana, text="Letra a recolectar (A-Z):").pack()
    entrada_letra = tk.Entry(ventana)
    entrada_letra.pack(pady=5)
    tk.Button(ventana, text="Recolectar Datos", command=pedir_letra).pack(pady=5)

    # Botón entrenar
    label_precision = tk.Label(ventana, text="Precisión del modelo: ---")
    label_precision.pack(pady=5)
    tk.Button(ventana, text="Entrenar Modelo",
              command=lambda: entrenar_modelo(label_precision)).pack(pady=5)

    # Botón reconocer
    tk.Button(ventana, text="Iniciar Reconocimiento en Tiempo Real",
              command=reconocer).pack(pady=10)

    ventana.mainloop()


# ================================
# EJECUCIÓN
# ================================
if __name__ == "__main__":
    interfaz()
