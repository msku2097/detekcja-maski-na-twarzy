# uzycie
# python3 detect_mask_video.py

# import paczek
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# znajdz wymiary ramki i zrob bloba z niej
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# przeslij boba przez siec i odbierz prawdopodobienstwo
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# zainicjuj naszą listę twarzy, odpowiadające im lokalizacje oraz listę prognoz z naszej sieci masek twarzy
	faces = []
	locs = []
	preds = []

	# loopuj przez iteracje
	for i in range(0, detections.shape[2]):
		# wyodrebnij prawdopodobienstwo 
		confidence = detections[0, 0, i, 2]

		# odfiltruj detekcje 
		if confidence > args["confidence"]:
			# narysuj kwadracik
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# upewnij sie ze kwadracik miesci sie w obszarze
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# wyodrębnij ROI twarzy, przekonwertuj go z BGR na kanał RGB, zmien rozmiar na 224x224 i wstępnie przetwórz
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# dodaj twarz i obrys do modelu
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# przewiduj tylko, jeśli wykryta zostanie co najmniej jedna twarz
	if len(faces) > 0:
		# dla szybszego wnioskowania dokonamy prognoz wsadowych na * wszystkich * twarzach w tym samym czasie, a nie jeden po drugim w powyższej pętli `for`
		preds = maskNet.predict(faces)

	# zwroc tupla z indexami twarzy
	return (locs, preds)


#skonstruuj parser argumentów i przeanalizuj argumenty
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="sciezka do modelu detekcji")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="sciezka do wytrenowanego modelu")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimalne prawdopodobieństwo odfiltrowania słabych detekcji")
args = vars(ap.parse_args())

#załaduj nasz zserializowany model detektora twarzy z dysku
print("[INFO] wczytuje model detekcji twarzy...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#załaduj model detektora maski na twarzy
print("[INFO] wczytuje model detekcji maski na twarzy...")
maskNet = load_model(args["model"])

# inicjalizujemy kamere
print("[INFO] startuje stream video")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loopoj po ramkach video
while True:
	#pobierz ramkę z wątkowego strumienia wideo i zmień jej rozmiar aby mieć maksymalną szerokość 400 pikseli
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

# wykryj twarze w ramce i sprawdź, czy mają na sobie maskę czy nie
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loopuj przez znalezione
	for (box, pred) in zip(locs, preds):
		# narysuj obwodke i predykcje
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# określ etykietę klasy i kolor, którego użyjemy do narysowania ramki i tekstu
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# dodaj prawdopodobieństwo na etykiecie
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# wyświetl etykietę i prostokąt ramki granicznej na obrazie
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# wyswietl przetworzone ramki
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# wcisnij 'q' aby zakonczyc
	if key == ord("q"):
		break

# sprzontanko
cv2.destroyAllWindows()
vs.stop()