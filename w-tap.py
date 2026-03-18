import pyautogui
import cv2
import numpy as np
import mss
import time

# Configuración de la región de pantalla (ajusta según tu ventana de Minecraft)
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

# Tiempo del W-tap (segundos)
wtap_duration = 0.05
wtap_interval = 0.05

# Color aproximado del enemigo (HSV) - esto es un ejemplo, puedes ajustar
# Para detectar enemigos con armadura roja
lower_color = np.array([20, 100, 100])
upper_color = np.array([35, 255, 255])

sct = mss.mss()

print("Bot W-tap iniciado. Presiona Ctrl+C para detener.")

try:
    while True:
        # Capturar pantalla
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detectar color
        mask = cv2.inRange(img, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Si se detecta enemigo
        if len(contours) > 0:
            # Presiona W (W-tap)
            pyautogui.keyDown('w')
            time.sleep(wtap_duration)
            pyautogui.keyUp('w')
            time.sleep(wtap_interval)

        # Opcional: ver la máscara para debug
        # cv2.imshow("Mask", mask)
        # if cv2.waitKey(1) == 27:
        #     break

except KeyboardInterrupt:
    print("Bot detenido.")
    cv2.destroyAllWindows()