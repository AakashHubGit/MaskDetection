# import the necessary packages

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.models import load_model

from imutils.video import VideoStream

import numpy as np

import imutils
import time

import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet,glassNet):

	(h, w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),

		(104.0, 177.0, 123.0))


	faceNet.setInput(blob)

	detections = faceNet.forward()


	faces = []

	locs = []

	preds_mask = []
	preds_glass = []


	for i in range(0, detections.shape[2]):


		confidence = detections[0, 0, i, 2]


		if confidence > 0.5:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			(startX, startY, endX, endY) = box.astype("int")


			(startX, startY) = (max(0, startX), max(0, startY))

			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			face = frame[startY:endY, startX:endX]

			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

			face = cv2.resize(face, (224, 224))

			face = img_to_array(face)

			face = preprocess_input(face)

			faces.append(face)

			locs.append((startX, startY, endX, endY))


	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds_glass = glassNet.predict(faces, batch_size=32)
		preds_mask = maskNet.predict(faces, batch_size=32)


	return (locs, preds_mask,preds_glass)


prototxtPath = r"face_detector\deploy.prototxt"

weightspath = r"face_detector\res_ssd_300Dim.caffeModel"


faceNet = cv2.dnn.readNet(prototxtPath, weightspath)


maskNet = load_model("mask_detector.h5")

glassNet = load_model("glasses_detector.h5")


print("Starting Video...")

vs = VideoStream(src=0).start()


while True:

    frame = vs.read()

    frame = imutils.resize(frame, width=800)


    (loc, pred_mask, pred_glass) = detect_and_predict_mask(frame, faceNet, maskNet, glassNet)


    for (box, pred_mask, pred_glass) in zip(loc, pred_mask, pred_glass):

        (startX, startY, endX, endY) = box

        (mask, withoutMask) = pred_mask
        (glass, withoutGlass) = pred_glass


        label_mask = "Mask" if mask > withoutMask else "No Mask"
        label_glass = "Glasses" if glass > withoutGlass else "No Glasses"
		

        color = (0,255,0) if label_mask=="No Mask" and label_glass=="Glasses" else (0,0,255)
        color = (255,0,0) if label_mask=="Mask" and label_glass=="No Glasses" else color


        label_mask = "{}: {:.2f}%".format(label_mask, max(mask, withoutMask) * 100)
        label_glass = "{}: {:.2f}%".format(label_glass, max(glass, withoutGlass) * 100)


        cv2.putText(frame, label_mask, (startX, startY - 10),

            cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.putText(frame, label_glass, (startX, endY + 10),

            cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

        cv2.rectangle(frame,(startX, startY), (endX, endY),color,2)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


cv2.destroyAllWindows()

vs.stop()

