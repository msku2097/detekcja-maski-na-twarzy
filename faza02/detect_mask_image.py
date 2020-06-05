# uzycie
# python3 detect_mask_image.py --image zdjecia/example_01.png

# import paczek
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# skonstruuj parser argumentów i przeanalizuj argumenty
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="sciezka do zdjecia")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="sciezka do modelu detekcji")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="sciezka do wytrenowanego modelu")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimalne prawdopodobieństwo odfiltrowania słabych detekcji")
args = vars(ap.parse_args())

# załaduj nasz zserializowany model detektora twarzy z dysku
print("[INFO] wczytuje model detekcji twarzy...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# załaduj model detektora maski na twarz z dysku
print("[INFO] wczytuje model detekcji maski na twarzy...")
model = load_model(args["model"])

# załaduj obraz wejściowy z dysku, sklonuj go i pobierz wymiary przestrzenne obrazu
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# skonstruuj blob z obrazu
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# wyslij bloba przez siec i odbierz predykcje
print("[INFO] obliczanie wykrycia twarzy...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# oblicz wspolrzedne ramki dla twarzy
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# upewnij sie, ze ramka miescie sie w rozmiarach obrazka
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# wyodrębnij ROI twarzy, przekonwertuj go z BGR na kanał RGB, zmien rozmiar na 224x224 i wstępnie przetwórz
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# przepuść twarz przez model, aby ustalić, czy twarz ma maskę czy nie
		(mask, withoutMask) = model.predict(face)[0]

 		#określ etykietę klasy i kolor, którego użyjemy do narysowania ramki i tekstu
		label = "Maska" if mask > withoutMask else "Bez maski"
		color = (0, 255, 0) if label == "Maska" else (0, 0, 255)

		# dodaj prawdopodobieństwo na etykiecie
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# wyświetl etykietę i prostokąt ramki granicznej na obrazie
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# wyswietl finalny obrazek
cv2.imshow("Output", image)
cv2.waitKey(0)