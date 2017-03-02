#!/usr/bin/env python

# Perception
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# Messages
from sensor_msgs.msg import Image

bridge = CvBridge()
img_points = [] 
ret_list = []

def image_callback(data):
	frame = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	# For annotation
	for i in range(len(img_points)):
		if (s=='A' or s=='a'):
			cv2.drawChessboardCorners(frame, (9,6), img_points[i] ,ret_list[i])
		elif (s=='B' or s=='b'):
			cv2.drawChessboardCorners(frame, (7,7), img_points[i] ,ret_list[i])
			
	cv2.imshow('capture', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		if (s=='A' or s=='a'):
			#Intereestingly, the funciton returns a false when it is applied twice to the same
			#image. Why?
			ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
			print ret
		elif (s=='B' or s=='b'):
			ret, corners = cv2.findCirclesGrid(gray, (7,7),None)
			print ret
		if (ret > 0):
			img_points.append(corners)
			ret_list.append(ret)
			file_name =	str(len(img_points)) + ".png"
			cv2.imwrite(file_name, gray)
			print "calibration image number: %d" % len(img_points)
	
def listener():
	rospy.init_node("kinect_calibration", anonymous=True)
	rospy.Subscriber("kinect2/hd/image_color", Image, image_callback)
	rospy.spin()

if "__main__" == __name__:
	try:
		print "Kinect Calibration Start"
		# Set calibration pattern
		print "Choose calibration pattern"
		print "A - Chessboard"
		print "B - Circular Grid"
		s = raw_input("--->")
		
		listener()
	except rospy.ROSInterruptException:
		pass
