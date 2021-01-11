#Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import cv2
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import time
import os
import pygame


class ShowVideo(QtCore.QObject):

	#initiating the built in camera
	camera_port = 0
	camera = cv2.VideoCapture(camera_port)
	VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
	last = 0

	def __init__(self, parent = None):
		super(ShowVideo, self).__init__(parent)
	
	def detect_and_predict_mask(self, frame, faceNet, maskNet):
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		faceNet.setInput(blob)
		detections = faceNet.forward()

		faces = []
		locs = []
		preds = []

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
			preds = maskNet.predict(faces, batch_size=32)

		return (locs, preds)

	@QtCore.pyqtSlot()
	def startVideo(self):

		run_video = True
		while run_video:
			ret, frame = self.camera.read()

			locs, preds = self.detect_and_predict_mask(frame, faceNet, maskNet)

			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred

				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

				if self.last > 40:
					if mask < withoutMask:
						pygame.mixer.music.play()
					else:
						pygame.mixer.music.pause()
					self.last = 0                   
				
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			self.last += 1
			color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			height, width, _ = color_swapped_image.shape
			
			qt_image = QtGui.QImage(color_swapped_image.data,
									width, 
									height,
			  						color_swapped_image.strides[0],
			   						QtGui.QImage.Format_RGB888)

			pixmap = QtGui.QPixmap(qt_image)
			qt_image = pixmap.scaled(1000, 800, QtCore.Qt.KeepAspectRatio)
			qt_image = QtGui.QImage(qt_image)

			self.VideoSignal.emit(qt_image)

class ImageViewer(QtWidgets.QWidget):
	def __init__(self, parent = None):
		super(ImageViewer, self).__init__(parent)
		self.image = QtGui.QImage()
		self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawImage(0,0, self.image)
		self.image = QtGui.QImage()

	def initUI(self):
		self.setWindowTitle('Test')
	
	@QtCore.pyqtSlot(QtGui.QImage)
	def setImage(self, image):
		if image.isNull():
			print("viewer dropped frame!")

		self.image = image
		if image.size() != self.size():
			self.setFixedSize(image.size())
		self.update()

# load our serialized face detector model
print("[INFO] loading face detector model...")
prototxtPath = "data/face_detector/deploy.prototxt"
weightsPath = "data/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("data/mask_detector.model")

# Initializing sound
print("[INFO] Initializing sound ...")
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('buzzer.wav')

if __name__ == '__main__':

	# GUI
	app = QtWidgets.QApplication(sys.argv)
	thread = QtCore.QThread()
	thread.start()

	vid = ShowVideo()
	vid.moveToThread(thread)
	image_viewer = ImageViewer()
	#image_viewer.resize(200,400)
	

	vid.VideoSignal.connect(image_viewer.setImage)
	
	#Button to start the videocapture:
	push_button = QtWidgets.QPushButton(QIcon('icons/start.png'), 'Start Video Stream')
	push_button.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
	push_button.clicked.connect(vid.startVideo)

	buzzer_button = QtWidgets.QPushButton(QIcon('icons/mute.png'), 'Stop Buzzer')
	buzzer_button.setStyleSheet('QPushButton{font: bold; font-size: 15pt; background-color: white;};')
	vertical_layout = QtWidgets.QVBoxLayout()

	vertical_layout.addWidget(image_viewer)
	vertical_layout.addWidget(push_button)
	vertical_layout.addWidget(buzzer_button)

	layout_widget = QtWidgets.QWidget()
	layout_widget.setLayout(vertical_layout)

	main_window = QtWidgets.QMainWindow()
	main_window.setCentralWidget(layout_widget)
	main_window.resize(1000,800)
	main_window.setWindowTitle("A+ Vision - Face Mask Detector")
	main_window.show()
	sys.exit(app.exec_())