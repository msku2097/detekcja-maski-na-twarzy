# użycie
# python3 train_mask_detector.py --dataset dataset

# import paczek
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# konstruktor argumentów 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="sciezka do danych")
ap.add_argument("-p", "--plot", type=str, default="graf.png",
	help="sciezka zapisu grafu")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="sciezka zapisu modelu mask detector")
args = vars(ap.parse_args())

#inicjalizacja szybkosci uczenia sie, liczbe epoch oraz wielkosc partii

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#Pobieramy listę zdjęć z naszego katalogu zestawu dantch, a nastepnie inicjalizujemy liste danych (obrazow) i obrazow klas
print("[INFO] wczytuję zdjecia...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop po sciezkach zdjec
for imagePath in imagePaths:
	# wyodrebniamy etykiete z nazwy pliku
	label = imagePath.split(os.path.sep)[-2]

	# ladujemy zdjecia (224x224) i procesujemy je
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# aktualizujemy listy danych i etykiet
	data.append(image)
	labels.append(label)

# konwertujemy dane i etykiety na tablice NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# encoding na labelkach
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# dzielimy dane na podzialy szkoleniowe i testowe przy uzyciu 75% danych szkolenia, a pozostale do testow

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# budujemy generator treningowy do obrazow 
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# ładujemy sieć MobileNetV2.
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# budujemy head modelu i zostanie umieszczona w trybier podstawowym
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# przekazanie modelu podstawowego na model rzeczywisty
model = Model(inputs=baseModel.input, outputs=headModel)

# loopujemy wszystkie warstwy na modelu podstawowym i mrozimy aktualizacje
for layer in baseModel.layers:
	layer.trainable = False

# kompilujemy model
print("[INFO] kompilujemy model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# trenujemy siec
print("[INFO] trenujemy siec...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# progozowanie na zestawie podstawowym
print("[INFO] predykcja sieci...")
predIdxs = model.predict(testX, batch_size=BS)

# dla każdego obrazu w zestawie testowym musimy znaleźć indeks i etykiete z odpowiednim największym przewidywanym prawdopodobieństwem
predIdxs = np.argmax(predIdxs, axis=1)

# wyswietl ladny raport klasyfikacji
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serializujemy model
print("[INFO] zapisuje model...")
model.save(args["model"], save_format="h5")

# wykres strat i dokładności treningu
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
#Funkcja straty, znana również jako funkcja kosztu, uwzględnia prawdopodobieństwo lub niepewność prognozy na podstawie tego, jak bardzo prognoza różni się od wartości rzeczywistej. To daje nam bardziej szczegółowy widok na wydajność modelu.
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Celnośc trenowanego modelu")
plt.xlabel("Epoch #")
plt.ylabel("strata/sprawnosc")
plt.legend(loc="lower left")
plt.savefig(args["plot"])