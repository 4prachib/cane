# import the necessary packages

from threading import Thread

import signal
import sys
import datetime
from math import floor
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import time
import cv2
import os
import RPi.GPIO as GPIO

#GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
		
class FrameCapture:
	def __init__(self, src=0):
		self.firstFrame = None
		self.camera = cv2.VideoCapture(src)
		while not self.camera.isOpened():
			self.camera = cv2.VideoCapture(src)	
			time.sleep(0.25)
		#print('Camera is ready...')
		(self.grabbed, self.frame) = self.camera.read()
		if self.grabbed:
			self.firstFrame = self.prepareFrame(self.frame)
		self.stopped = False

	def start(self):
		# Start capturing in new Thread.
		t = Thread(target=self.capture, args=())
		t.daemon = True
		t.start()
		return self

	def capture(self): 
		while(not self.stopped):
			(self.grabbed, self.frame) = self.camera.read()
		
		#print('Releasing camera')
		self.camera.release()

	# Read the captured frame as-is.
	def readFrame(self):
		# resize this frame, convert it to grayscale, and blur it
		self.frame = imutils.resize(self.frame, height=240)
		return self.frame	
 
	def prepareFrame(self, frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#gray = cv2.GaussianBlur(gray, (21,21), 0)	
		return gray

	def stop(self):
		self.stopped = True

class Sensor:
	# First sensor is pointing downwards and second detects frontal obstructions.
	# static variable.
	sensors = [{'name': 'Down firing', 'trig': 16, 'echo': 18}, {'name': 'Frontal', 'trig': 13, 'echo': 15}]
	max_observations = 3

	def __init__(self, sensor_id=0):
		#print('Turning sensor %d on...' % sensor_id)
		self.sensor = Sensor.sensors[sensor_id]

		GPIO.setup(self.sensor['trig'],GPIO.OUT)
		GPIO.setup(self.sensor['echo'],GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
		GPIO.output(self.sensor['trig'], False)
		time.sleep(1)
		#print('Sensor %d is ready...' % sensor_id)

		self.pulse_start = 0
		self.pulse_end = 0
		self.object_distance = 0
		
		# Observations for obstacles. We take 3 observations and compute average distance.
		self.observations = []
		self.observation_count = 0

		self.stopped = False

	def start(self):
		# Start processing sensor input in new Thread.
		t = Thread(target=self.capture, args=())
		t.daemon = True
		t.start()
		#self.capture()
		return self

	def fire(self):
		GPIO.output(self.sensor['trig'], False)
		time.sleep(0.1)
		GPIO.output(self.sensor['trig'], True)
		time.sleep(0.00001)
		GPIO.output(self.sensor['trig'], False)
		self.pulse_start = time.time()

	def capture(self): 
		while not self.stopped:
			self.fire();
			#print ('waiting for echo 1 %s'% self.sensor['name'])

			while GPIO.input(self.sensor['echo']) == 0:
		 		self.pulse_start = time.time()

			while GPIO.input(self.sensor['echo']) == 1:
		 		self.pulse_end = time.time()
		
			object_distance = (self.pulse_end - self.pulse_start) * 17150
			self.observation_count += 1
			if self.observation_count > Sensor.max_observations:
				average = sum(self.observations) / float(len(self.observations))
				del self.observations[:]
				self.observation_count = 0;
				# There is at least 5 cm movement since last value
				# print('%f -- %f' % (average, self.object_distance))
				if abs(average - self.object_distance) > 5:
					self.object_distance = average
			else:
				self.observations.append(object_distance)

	# Read the measured distance from last pulse.
	def read(self):
		#print ('Sensor %s -- distance %f' %(self.sensor['name'], self.object_distance))
		return self.object_distance	

	def stop(self):
		self.stopped = True

class ObstacleWarning:
	# Class variables
	warning_distance = 240 	# 8 ft
	crowd_size = 5		# What defines "crowd"?
	hush_period = 0.1	# Minimum period between warnings.

	def __init__(self):
		# Queue to maintain all objectacles detected and all uneven surface detection
		# Two separate queues are maintained since they both are important to warn the user 
		
		self.obstacle_warning_queue = {'last_update_time': 0, 'queue': []}
		self.surface_warning_queue = {'last_update_time': 0, 'queue': []}

		# All system messages.
		self.system_queue = {'last_update_time': 0, 'queue': []}

		# Prepare camera.
		self.camera = FrameCapture(src=0).start()
		self.add_to_system_queue('camera ready')
		# Current Frame being processed.
		self.frame = None

		# Prepare the sensors.
		self.front_sensor_last_reading = 0
		self.down_firing_sensor_readings = []
		self.down_firing_sensor_last_reading = 0
		self.down_firing_sensor = Sensor(sensor_id=0).start()
		self.front_sensor = Sensor(sensor_id=1).start()

		# People detector.
		self.hog = cv2.HOGDescriptor()
		self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

		# Is this stopped?
		self.stopped = False

	def hog_detection(self):
		#start_time = datetime.datetime.now()
		frame = self.frame		
		(rects, weights) = self.hog.detectMultiScale(frame.copy(), winStride=(8, 8),padding=(24, 24), scale=1.05)

		# draw the original bounding boxes
		#for (x, y, w, h) in rects:
		#	cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 			
		number_of_people = 0
		number_of_people_in_warning_zone = 0

		# Distance in cm of any obstruction detected by front sensor.
		distance = self.front_sensor.read()
		if self.front_sensor_last_reading == distance:
			return
		self.front_sensor_last_reading = distance

		frame_height = np.size(self.frame, 0)
		for (xA, yA, xB, yB) in pick:
			#distance = (focal_length * height_of_person * np.size(frame, 0)) / ((yB-yA) *2.48)
			#distance = distance *.67
			#distance = 408.7*62/(yB-yA)

			# If the people in the frame are "too far" ignore them. If the "height"
			# of rect is smaller than 1/4 of frame height, it is ignored.
			if (yB-yA) >= frame_height/4:
				number_of_people_in_warning_zone += 1

			# This is used to determine if we are in a crowd.
			number_of_people += 1
		 
			#cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

		if number_of_people_in_warning_zone > 0:
			if distance <= ObstacleWarning.warning_distance:
				text = 'crowd' if number_of_people > ObstacleWarning.crowd_size else 'people' 
				# Just queue the warning.
				self.obstacle_warning_queue['queue'].append('%s at %d feet' % (text, floor(distance/(12*2.54)))) 
		elif number_of_people > ObstacleWarning.crowd_size:
			self.obstacle_warning_queue['queue'].append('You are in a crowded place')
		else:
			# Not people. See if there are other obstacles such as objects, etc.
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (21,21), 0)
			edged = cv2.Canny(gray, 35, 125)
			#thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
			#thresh = cv2.dilate(thresh, None, iterations=2)
			_, cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			number_of_objects = 0
			# loop over the contours
			for c in cnts:
				(x, y, w, h) = cv2.boundingRect(c)
				#if the contour is too small, ignore it
				#area = cv2.contourArea(c)

				#cv2.rectangle(frame, (x, y,), (x + w, y + h), (0, 255, 0), 2)
				if h >= frame_height*.2 and h <= frame_height*0.75:
					#cv2.rectangle(frame, (x, y,), (x + w, y + h), (0, 255, 0), 2)
					number_of_objects = number_of_objects + 1
			if number_of_objects > 0 and distance > 30 and distance < 240:
				self.obstacle_warning_queue['queue'].append('obstacles at %d feet'% floor (distance/(12*2.54)))
		#print('Took %d' %  (datetime.datetime.now() - start_time).total_seconds())

	def calibrate_down_firing_sensor(self):
		calibration = []
		while len(calibration) < 100:
			calibration.append(self.down_firing_sensor.read())
		# calibrated distance based on person's height, sensor angle, etc.
		self.down_firing_sensor_last_reading = sum(calibration) / float(len(calibration))
		self.add_to_system_queue('calibrated distance %d centimeters' % floor(self.down_firing_sensor_last_reading))

	def uneven_surface_detection(self):
		if not self.down_firing_sensor_last_reading:
			return self.calibrate_down_firing_sensor()

		self.down_firing_sensor_readings.append(self.down_firing_sensor.read())
		if len(self.down_firing_sensor_readings) > 10:
			# Retire older values.
			self.down_firing_sensor_readings.pop(0)
			average = sum(self.down_firing_sensor_readings) / float(len(self.down_firing_sensor_readings))
			#print ('Average %f %f' % (average, self.down_firing_sensor_last_reading))
			if (average - self.down_firing_sensor_last_reading) > 15:
				self.surface_warning_queue['queue'].append('Uneven surface')
				
	def process(self):
		# Create a thread for voice outputs, so it is on its own.
		t = Thread(target=self.say, args=())
		t.daemon = True
		t.start()

		while not self.stopped:
			self.frame = self.camera.readFrame()

			self.hog_detection()
			self.uneven_surface_detection()
		
			# draw the text and timestamp on the frame
			#cv2.putText(frame, "Room Status: {}".format(text), (10,20),
				#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			#cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
				#(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

			# show the frame and record if the user presses a key
			#cv2.imshow("Security Feed", self.frame)
			#cv2.imshow("Thresh", thresh)
			#cv2.imshow("Frame Delta", frameDelta)
			#key = cv2.waitKey(1) & 0xFF

	def add_to_system_queue(self, text):
		self.system_queue['queue'].append(text)

	# Uses text-to-speech to say something. 
	def say(self):
		def runPico(text):
			os.system('pico2wave -w text.wav '+'"'+text+'"'+ '&& omxplayer -o local text.wav')

		while True:
			# Preference is for surface warning as it warns immediate obstacle.
			warned = False
			if len(self.surface_warning_queue['queue']):
				text = self.surface_warning_queue['queue'].pop()
				# Delete older warnings.
				del self.surface_warning_queue['queue'][:]
				runPico(text)
				warned = True
			if len(self.obstacle_warning_queue['queue']):
				now = time.time ()
				# Don't annoy the user repeatedly if the warnings come too quickly.
				if now - self.obstacle_warning_queue['last_update_time'] > ObstacleWarning.hush_period:
					text = self.obstacle_warning_queue['queue'].pop()
					del self.obstacle_warning_queue['queue'][:]
					runPico(text)
					# Delete older warnings.
					self.obstacle_warning_queue['last_update_time'] = now
					warned = True
			if len(self.system_queue['queue']):
				text = self.system_queue['queue'].pop()
				runPico(text)
				warned = True

			if not warned:
				# If warned, there is sufficient delay in processing next message.
				time.sleep(0.1)

	def stop(self):
		self.stopped = True
		self.camera.stop()
		self.down_firing_sensor.stop()
		self.front_sensor.stop()
		cv2.destroyAllWindows()

	def handle_ctrl_c(signal, frame):
		#print ('CTRL+C received')
		self.stop()
		sys.exit(0)
		
if __name__ == '__main__':
	warner = ObstacleWarning().process()	
	signal.signal(signal.SIGINT, warner.handle_ctrl_c)
	warner.add_to_system_queue('system ready')
	
	
