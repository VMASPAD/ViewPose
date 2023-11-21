# Importamos las librerías
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("POSE.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(1)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    
    # Leemos resultados
    resultados = model.predict(frame, imgsz=340)
    
    # Mostramos resultados
    anotaciones = resultados[0].plot()
    
    # Mostramos nuestros fotogramas
    cv2.imshow("View", anotaciones)
    
    

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
