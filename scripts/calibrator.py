import cv2
import numpy as np

class Calibrator:
	
	pattern_type = "CHESS_BOARD"
	calibration_file = "calibration_s4/"
	debug = False
	calibrated = False
	global mtx
	global dist
	global reproj_err
	
	def __init__(self):
		pass
	
	def set_pattern_type(self, s):
		pattern_type = s
		
	def set_calibration_file(self, s):
		calibration_file = s
		
	def set_debug(self):
		self.debug = True
	
	def perform_calibration(self):
		
		print "Begin Calibration"
		
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		success = 0

		#===========================STEP 1: PREPARE CALIBRATION PARAMETERS===========================
		print "Step 1: Preparing parameters"
		
		if(self.pattern_type=="CHESS_BOARD"):
			objp = np.zeros((6*9,3), np.float32)
			objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
		elif(self.pattern_type=="CIRCLE_GRID"):
			pattern_width = 7
			pattern_height = 7
			#~ pattern_separation = 0.01594
			pattern_separation = 0.0185
			objp = np.zeros((pattern_width*pattern_height,3), np.float32)
	
			for i in range(pattern_width):
				for j in range(pattern_height):
					objp[i*pattern_height+j,0] = pattern_separation*i
					objp[i*pattern_height+j,1] = pattern_separation*j
					objp[i*pattern_height+j,2] = 0

		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.

		for n in range(10):
			print "Loading frame %d" % n
			
			file_name = self.calibration_file + str(n) + ".jpg"
			frame = cv2.imread(file_name)
			
			if frame is None:
				break
				
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			
			# Find the chess board corners
			if(self.pattern_type=="CHESS_BOARD"):
				ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
			elif(self.pattern_type=="CIRCLE_GRID"):
				ret, corners = cv2.findCirclesGrid(gray, (7,7),None)
				
			# If found, add object points, image points (after refining them)
			if ret == True:
					objpoints.append(objp)
					cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
					imgpoints.append(corners)
					
					if self.debug==True:
						# Draw and display the corners
						if(self.pattern_type=="CHESS_BOARD"):
							cv2.drawChessboardCorners(frame, (9,6), corners ,ret)
						elif(self.pattern_type=="CIRCLE_GRID"):
							cv2.drawChessboardCorners(frame, (7,7), corners ,ret)
						cv2.imshow('img',frame)
						cv2.waitKey(0)
						
					success = success + 1
						
			n=n+1

		#===========================STEP 2: PERFORM CALIBRATION===========================
		print "Number of successful pattern detection: %d" 5 success
		
		if (success > 5):
			print "Step 2: Begin calibration"
			self.reproj_err, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
			self.calibrated = True
	
	def print_parameters(self):
			print "\nCamera matrix: "
			print self.mtx
			print "\nDistortion matrix: "
			print self.dist
			print "\nRe-projection error: "
			print self.reproj_err
