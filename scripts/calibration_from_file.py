#!/usr/bin/env python

from calibrator import Calibrator
import rospy
calibrator = Calibrator()

if "__main__" == __name__:
	try:
		calibrator.set_pattern_type("CHESS_BOARD")
		calibrator.set_calibration_file("calibration_sd_kinect/")
		#~ calibrator.set_calibration_file("calibration_s4/")
		calibrator.perform_calibration()
		calibrator.print_parameters()
	except rospy.ROSInterruptException:
		pass
