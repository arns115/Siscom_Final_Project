#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traductor LSM mejorado: letras -> palabras -> voz
Archivo: traductor_lsm.py

Requisitos (instalar con pip si falta):
pip install opencv-python mediapipe pandas scikit-learn joblib pyttsx3

Uso:
- Ejecuta el script.
- En la GUI: recolecta letra por letra para crear dataset (o usa dataset existente).
- Entrena el modelo.
- Inicia reconocimiento: hará buffering de letras, detectará palabras por pausa,
  las corregirá automáticamente con el diccionario y las pronunciará.
"""

import cv2
import csv
import os
import time
import threading
from collections import deque
from datetime import datetime
import difflib

import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, filedialog

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pyttsx3

# ================================
# CONFIG
# ================================
DATASET = "abecedario.csv"
MODELO = "modelo_abecedario.joblib"
LOG_LETRAS = "bitacora_predicciones.csv"
LOG_PALABRAS = "bitacora_palabras.csv"
DICCIONARIO_ARCHIVO = "diccionario.txt"  # opcional: carga palabras si existe

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Voz
voz = pyttsx3.init()
voz.setProperty("rate", 150)

# Parámetros de reconocimiento/ensamblado
STABILITY_FRAMES = 6      # cuántos frames iguales para aceptar letra
PROB_THRESHOLD = 0.55     # probabilidad mínima del clasificador
PRED_WINDOW = 10          # ventana para voto mayoritario (frames)
WORD_GAP_SECONDS = 1.0    # tiempo sin mano para considerar fin de palabra
MAX_WORD_LENGTH = 50

# Estado compartido (controlado por GUI / hilos)
recognition_thread = None
recognizing = False

# ================================
# Diccionario de palabras (común, para autocorrección)
# Primero intentamos cargar DICCIONARIO_ARCHIVO si existe,
# sino usamos una lista básica de palabras comunes en español.
# ================================
def cargar_diccionario():
    if os.path.exists(DICCIONARIO_ARCHIVO):
        with open(DICCIONARIO_ARCHIVO, "r", encoding="utf-8") as f:
            palabras = [line.strip() for line in f if line.strip()]
        return list(dict.fromkeys(palabras))  # dedup
    # Lista básica (puedes ampliarla o proveer diccionario.txt)
    base = """
    hola adios gracias por favor si no yo tu el la los las un una en con para como
    bien mal hoy mañana ayer casa escuela trabajo ayuda necesito quiero puedes
    quieres donde cuando quien que esto eso ella elos amigo amiga salud nombre
    venir ir venir volver decir hablar escuchar mano letra palabra palabra
    bueno bueno noche dia claro gracias amigo familia amor tiempo hombre mujer
    niño niña coche autobus tren aeropuerto abrir cerrar escribir leer aprender
    prueba probar ejemplo mensaje telefono correo saludos
    """
    palabras = [w.strip() for w in base.split() if w.strip()]
    return sorted(set(palabras))

DICCIONARIO = cargar_diccionario()

# ================================
# UTIL: extraer landmarks (63 valores)
# ================================
def extraer_landmarks(results):
    """Extrae 63 valores (21 puntos * XYZ) de la primera mano detectada."""
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    coords = []
    for p in lm.landmark:
        coords.extend([p.x, p.y, p.z])
    return coords  # lista de 63 floats

# ================================
# RECOLECCIÓN DE DATOS
# ================================
def iniciar_recoleccion(letra, muestras=250):
    """Abre cámara y recolecta 'muestras' de la letra indicada, guardando en DATASET."""
    letra = letra.upper()
    if len(letra) != 1 or not letra.isalpha():
        raise ValueError("Letra inválida")

    # Crear dataset si no existe
    if not os.path.exists(DATASET):
        with open(DATASET, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            columnas = ["label"] + [f"v{i}" for i in range(21*3)]
            writer.writerow(columnas)

    cap = cv2.VideoCapture(0)
    manos = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)

    total = 0
    mensaje = f"Recolectando letra {letra}: {total}/{muestras}"
    last_save_time = 0

    while total < muestras:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = manos.process(rgb)

        feat = extraer_landmarks(results)
        if feat is not None:
            # Guardar con una pequeña frecuencia para evitar duplicados rápidos
            now = time.time()
            if now - last_save_time > 0.05:
                with open(DATASET, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([letra] + feat)
                total += 1
                last_save_time = now

        # Mostrar feedback en la ventana de OpenCV
        cv2.putText(frame, f"Recolectando: {letra} ({total}/{muestras})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Recolección LSM - Presiona 'q' para cancelar", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    manos.close()

# ================================
# ENTRENAMIENTO
# ================================
def entrenar_modelo():
    """Entrena RandomForest con los datos en DATASET y guarda en MODELO."""
    if not os.path.exists(DATASET):
        raise FileNotFoundError("No existe dataset. Recolecta datos primero.")

    df = pd.read_csv(DATASET)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Verificar balance mínimo
    counts = df['label'].value_counts()
    if counts.min() < 10:
        # aún se puede entrenar, pero advertir
        print("Advertencia: algunas clases tienen pocas muestras. Mejora el dataset para mejor rendimiento.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.18, stratify=y, random_state=42
    )

    modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, pred)

    joblib.dump(modelo, MODELO)
    return precision

# ================================
# LOGS / BITÁCORAS
# ================================
def registrar_letra(letra):
    existe = os.path.exists(LOG_LETRAS)
    with open(LOG_LETRAS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["Fecha", "Letra"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), letra])

def registrar_palabra(palabra):
    existe = os.path.exists(LOG_PALABRAS)
    with open(LOG_PALABRAS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["Fecha", "Palabra", "Palabra_corregida"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), palabra, corregir_palabra(palabra)])

# ================================
# AUTOCORRECCIÓN
# ================================
def corregir_palabra(palabra):
    """Corrige palabra usando difflib y DICCIONARIO. Devuelve la mejor coincidencia o la palabra original."""
    palabra = palabra.lower()
    if not palabra:
        return palabra
    # buscar entre diccionario
    matches = difflib.get_close_matches(palabra, DICCIONARIO, n=1, cutoff=0.7)
    if matches:
        return matches[0]
    # alternativa: tomar palabra tal cual si no encontró
    return palabra

# ================================
# RECONOCIMIENTO EN TIEMPO REAL (hilo)
# ================================
def reconocimiento_loop(gui_callbacks):
    """
    Bucle de reconocimiento que:
    - lee camera
    - detecta landmarks
    - vota/estabiliza letra
    - arma palabras por pausas
    - corrige y habla palabra
    gui_callbacks: dict con callbacks para actualizar GUI:
        - actualizar_letra(letra)
        - actualizar_palabra(palabra)
    """
    global recognizing
    if not os.path.exists(MODELO):
        gui_callbacks['mostrar_error']("Error", "No existe modelo entrenado. Entrena primero.")
        recognizing = False
        return

    modelo = joblib.load(MODELO)
    cap = cv2.VideoCapture(0)
    manos = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)

    pred_queue = deque(maxlen=PRED_WINDOW)  # últimas predicciones
    stable_queue = deque(maxlen=STABILITY_FRAMES)  # para detectar estabilidad exacta
    last_hand_time = time.time()
    current_word = ""
    last_accepted_letter_time = 0

    while recognizing:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = manos.process(rgb)

        feat = extraer_landmarks(results)
        predicted_letter = ""
        accepted = False

        if feat is not None:
            last_hand_time = time.time()
            X = np.array(feat).reshape(1, -1)

            # predicción y prob
            try:
                probs = modelo.predict_proba(X)[0]
                classes = modelo.classes_
                idx = np.argmax(probs)
                letra_hat = classes[idx]
                prob = probs[idx]
            except Exception:
                # En caso de que predict_proba no esté disponible
                letra_hat = modelo.predict(X)[0]
                prob = 1.0

            pred_queue.append(letra_hat)
            stable_queue.append(letra_hat)

            # voto mayoritario en ventana
            voto = max(set(pred_queue), key=pred_queue.count)
            # comprobar estabilidad: última N frames iguales
            if len(stable_queue) == STABILITY_FRAMES and len(set(stable_queue)) == 1:
                # y que la probabilidad sea suficiente
                if prob >= PROB_THRESHOLD and voto == letra_hat:
                    predicted_letter = letra_hat
                    # prevenir letras repetidas por frames continuos: solo aceptar si pasó suficiente tiempo desde la última aceptación
                    now = time.time()
                    if now - last_accepted_letter_time > 0.4:  # debounce en segundos
                        accepted = True
                        last_accepted_letter_time = now
        else:
            pred_queue.append("")  # vacío cuando no hay mano

        # Si se acepta una letra, añadir al buffer de palabra
        if accepted and predicted_letter:
            current_word += predicted_letter
            registrar_letra(predicted_letter)
            gui_callbacks['actualizar_letra'](predicted_letter)
            gui_callbacks['actualizar_palabra'](current_word)
            # no decir letra individualmente; esperamos a palabra final para pronunciar
        else:
            # mostrar la letra "tentativa" en GUI aunque no esté aceptada
            if feat is not None:
                # mostrar voto mayoritario como tentativo
                voto = max(set(pred_queue), key=pred_queue.count) if pred_queue else ""
                gui_callbacks['actualizar_letra'](voto)
            else:
                gui_callbacks['actualizar_letra']("")

        # Dibujo y ventana OpenCV (feedback)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        # mostrar word y letra encima de frame
        cv2.rectangle(frame, (0,0), (400,90), (0,0,0), -1)
        txt_letter = f"Letra: {predicted_letter if predicted_letter else ''}"
        txt_word = f"Palabra: {current_word}"
        cv2.putText(frame, txt_letter, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, txt_word, (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Reconocimiento LSM - Presiona 'q' para detener", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recognizing = False
            break

        # DETECCIÓN FIN DE PALABRA: si no se detecta mano por WORD_GAP_SECONDS
        now = time.time()
        if now - last_hand_time > WORD_GAP_SECONDS and current_word:
            # finaliza palabra
            palabra_original = current_word
            palabra_corregida = corregir_palabra(palabra_original)
            gui_callbacks['actualizar_palabra'](palabra_corregida)
            # hablar palabra corregida
            voz.say(palabra_corregida)
            voz.runAndWait()
            registrar_palabra(palabra_original)
            current_word = ""  # reset
            gui_callbacks['actualizar_letra']("")  # limpiar letra en GUI
            gui_callbacks['actualizar_palabra'](current_word)

        # prevenir overflow de palabra
        if len(current_word) > MAX_WORD_LENGTH:
            current_word = current_word[:MAX_WORD_LENGTH]

    # limpieza al acabar
    cap.release()
    cv2.destroyAllWindows()
    manos.close()
    recognizing = False
    gui_callbacks['on_recognition_stop']()

# ================================
# GUI (Tkinter)
# ================================
class TraductorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traductor LSM - Letras a Palabras")
        self.root.geometry("520x360")

        # Frame superior
        tk.Label(root, text="Traductor LSM: Abecedario → Palabras", font=("Arial", 14)).pack(pady=8)

        # Entrada para recolección
        frm = tk.Frame(root)
        frm.pack(pady=4)
        tk.Label(frm, text="Letra (A-Z):").grid(row=0, column=0, padx=4)
        self.entrada_letra = tk.Entry(frm, width=4, font=("Arial", 14))
        self.entrada_letra.grid(row=0, column=1, padx=4)
        tk.Button(frm, text="Recolectar (250 muestras)", command=self.cmd_recolectar).grid(row=0, column=2, padx=6)

        # Entrenamiento
        tk.Button(root, text="Entrenar modelo", command=self.cmd_entrenar).pack(pady=6)
        self.label_precision = tk.Label(root, text="Precisión: ---")
        self.label_precision.pack()

        # Estado reconocimiento y display
        frm2 = tk.Frame(root)
        frm2.pack(pady=8)
        tk.Button(frm2, text="Iniciar reconocimiento", command=self.cmd_iniciar_reconocimiento).grid(row=0, column=0, padx=6)
        tk.Button(frm2, text="Detener reconocimiento", command=self.cmd_detener_reconocimiento).grid(row=0, column=1, padx=6)

        # Información en tiempo real
        tk.Label(root, text="Letra detectada (tentativo/aceptado):").pack()
        self.label_letra = tk.Label(root, text="", font=("Arial", 20), fg="blue")
        self.label_letra.pack()

        tk.Label(root, text="Palabra en construcción:").pack()
        self.label_palabra = tk.Label(root, text="", font=("Arial", 18), fg="green")
        self.label_palabra.pack()

        # Botones de control de palabra
        frm3 = tk.Frame(root)
        frm3.pack(pady=8)
        tk.Button(frm3, text="Borrar palabra", command=self.cmd_borrar_palabra).grid(row=0, column=0, padx=6)
        tk.Button(frm3, text="Confirmar y hablar palabra", command=self.cmd_confirmar_palabra).grid(row=0, column=1, padx=6)
        tk.Button(frm3, text="Cargar diccionario...", command=self.cmd_cargar_diccionario).grid(row=0, column=2, padx=6)

        # Estado interno
        self.current_display_word = ""
        self.current_display_letter = ""

        # Callbacks para reconocimiento
        self.gui_callbacks = {
            'actualizar_letra': self.actualizar_letra,
            'actualizar_palabra': self.actualizar_palabra,
            'mostrar_error': self.mostrar_error,
            'on_recognition_stop': self.on_recognition_stopped
        }

    # -------------------------------
    # Comandos GUI
    # -------------------------------
    def cmd_recolectar(self):
        letra = self.entrada_letra.get().strip().upper()
        if len(letra) != 1 or not letra.isalpha():
            messagebox.showerror("Error", "Ingresa una sola letra A-Z.")
            return
        # Recolección en hilo para no bloquear GUI
        t = threading.Thread(target=self._hilo_recoleccion, args=(letra,), daemon=True)
        t.start()

    def _hilo_recoleccion(self, letra):
        try:
            iniciar_recoleccion(letra, muestras=250)
            messagebox.showinfo("Recolección", f"Recolección de {letra} finalizada.")
        except Exception as e:
            messagebox.showerror("Error Recolección", str(e))

    def cmd_entrenar(self):
        # Entrenamiento en hilo
        t = threading.Thread(target=self._hilo_entrenar, daemon=True)
        t.start()

    def _hilo_entrenar(self):
        try:
            precision = entrenar_modelo()
            self.label_precision.config(text=f"Precisión: {precision*100:.2f}%")
            messagebox.showinfo("Entrenamiento", f"Modelo entrenado con precisión {precision*100:.2f}%")
        except Exception as e:
            messagebox.showerror("Error Entrenamiento", str(e))

    def cmd_iniciar_reconocimiento(self):
        global recognizing, recognition_thread
        if recognizing:
            messagebox.showinfo("Info", "Reconocimiento ya está en ejecución.")
            return
        recognizing = True
        recognition_thread = threading.Thread(target=reconocimiento_loop, args=(self.gui_callbacks,), daemon=True)
        recognition_thread.start()
        messagebox.showinfo("Reconocimiento", "Reconocimiento iniciado. Cierra la ventana OpenCV o presiona 'q' para detenerlo.")

    def cmd_detener_reconocimiento(self):
        global recognizing
        if not recognizing:
            messagebox.showinfo("Info", "No hay reconocimiento en ejecución.")
            return
        recognizing = False

    def cmd_borrar_palabra(self):
        # Borra palabra actual (nota: reconocimiento la seguirá construyendo)
        self.current_display_word = ""
        self.label_palabra.config(text=self.current_display_word)

    def cmd_confirmar_palabra(self):
        # Confirmar manual: corregir, hablar y registrar
        palabra = self.current_display_word
        if not palabra:
            messagebox.showinfo("Info", "No hay palabra para confirmar.")
            return
        palabra_corr = corregir_palabra(palabra)
        registrar_palabra(palabra)
        voz.say(palabra_corr)
        voz.runAndWait()
        messagebox.showinfo("Palabra confirmada", f"Original: {palabra}\nCorregida: {palabra_corr}")
        self.current_display_word = ""
        self.label_palabra.config(text="")

    def cmd_cargar_diccionario(self):
        path = filedialog.askopenfilename(title="Selecciona archivo diccionario (lineas con palabras)",
                                          filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                palabras = [line.strip() for line in f if line.strip()]
            global DICCIONARIO
            DICCIONARIO = sorted(set(palabras))
            messagebox.showinfo("Diccionario", f"Cargado {len(DICCIONARIO)} palabras.")
        except Exception as e:
            messagebox.showerror("Error", f"No fue posible cargar diccionario: {e}")

    # -------------------------------
    # Callbacks que usa el hilo de reconocimiento
    # -------------------------------
    def actualizar_letra(self, letra):
        # actualiza la letra en la GUI (se llama desde hilo)
        self.current_display_letter = letra
        try:
            self.label_letra.config(text=letra)
        except tk.TclError:
            pass

    def actualizar_palabra(self, palabra):
        self.current_display_word = palabra
        try:
            self.label_palabra.config(text=palabra)
        except tk.TclError:
            pass

    def mostrar_error(self, titulo, msg):
        messagebox.showerror(titulo, msg)

    def on_recognition_stopped(self):
        # llamado cuando termina el hilo
        messagebox.showinfo("Reconocimiento", "Reconocimiento detenido.")

    def on_recognition_stopped(self):
        # llamada segura al detener reconocimiento
        try:
            messagebox.showinfo("Reconocimiento", "Reconocimiento detenido.")
        except tk.TclError:
            pass

# ================================
# EJECUCIÓN
# ================================
def main():
    root = tk.Tk()
    app = TraductorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
