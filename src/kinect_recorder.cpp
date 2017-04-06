// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <ros/callback_queue.h>
#include <yaml-cpp/yaml.h>
#include <rosbag/bag.h>

// FOR IMAGE RECORDING
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "data_manager.cpp"

// IMAGE PROCESSING
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>
// POINT CLOUD PROCESSING
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>

DataManagement dm;
std::string package_path_;
std::string save_path;
std::string bag_name;

bool active_threads = true;

void *start_cv_main(void *threadid){
	cv::VideoCapture inputVideo;
	inputVideo.open(0);
	bool marker_found;
	while (inputVideo.grab() && active_threads) {
		//~ std::cout << "Grabbing Frame" << std::endl;
		cv::Mat image;
		inputVideo.retrieve(image);
		dm.loadFrame(image);
	}
	std::cout << "Exiting Cv Main" << std::endl;
	pthread_exit(NULL);
}

void *image_recorder(void *threadid){
	cv::namedWindow("Image Viewer",1);
	cv_bridge::CvImage out_msg;
	
	rosbag::Bag bag;
	bag.open(save_path + bag_name, rosbag::bagmode::Write);
	
	while (active_threads){
		cv::Mat image;
		if (dm.getFrame(image)){
			// VISUALISATION FOR DEBUGGING
			cv::imshow("Image Viewer", image);
			if(cv::waitKey(30) >= 0){
				//~ std::cout << "Key out" << std::endl;
				active_threads = false;
			}
			
			// CONVERSION TO ROS MSG FOR SAVE
			out_msg.encoding = sensor_msgs::image_encodings::BGR8;
			out_msg.image    = image;
			
			// WRITING TO ROSBAG
			bag.write("image", ros::Time::now(), out_msg.toImageMsg());
		}
	}
	active_threads = false;
	bag.close();
	
	std::cout << "Exiting Image Viewer" << std::endl;
}

int main (int argc, char** argv){
	std::cout << std::endl << "Kinect Record Started" << std::endl;
  ros::init(argc, argv, "kinect_rrecord");
    
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  save_path = package_path_ + "/kinect_record_saves/";
  std::string point_cloud_topic_name = "/kinect2/sd/points";
  std::string image_topic_name = "/kinect2/sd/image_color_rect";
  
	// LOAD SINGLE IMAGE AND POINT CLOUD
	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_("~");
	double begin =ros::Time::now().toSec();
	double current = ros::Time::now().toSec();
	double time_limit = 5.0;
	nh_private_.getParam("bag_name_", bag_name);
	
	// THREAD INITIALISATION	
	pthread_t thread[2];
	int threadError;
	int i=0;
	
	// START CV THREAD
	threadError = pthread_create(&thread[i], NULL, start_cv_main, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}
	i++;
	
	threadError = pthread_create(&thread[i], NULL, image_recorder, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}

	pthread_exit(NULL);
	
	ros::shutdown();	
	
	std::cout << "Exiting Main Thread" << std::cout;
	return 0;
}
