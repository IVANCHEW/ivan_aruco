#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>

// FOR IMAGE STUFF
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// FOR OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

std::string package_path_;
std::string folder_path_;
std::string bag_name_;

int main (int argc, char** argv){

	std::cout << std::endl << "Rosbag Replay Started" << std::endl;
  ros::init(argc, argv, "rosbag_replay");
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_("~");
  
  nh_private_.getParam("bag_name_", bag_name_);
  nh_private_.getParam("folder_name_", folder_path_);
  
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  
  // VISUALISATION FOR DEBUGGING
  cv::namedWindow("Image Viewer",1);
  
	rosbag::Bag bag;
	bag.open(package_path_ + folder_path_ + bag_name_, rosbag::bagmode::Read);
	
	std::vector<std::string> topics;
	topics.push_back(std::string("image"));
	
	rosbag::View view(bag, rosbag::TopicQuery(topics));
	BOOST_FOREACH(rosbag::MessageInstance const m, view)
	{
		std::cout << "Accessing Message" << std::endl;
		cv::Mat image;
		sensor_msgs::Image::ConstPtr msg = m.instantiate<sensor_msgs::Image>();
		std::cout << "Message retrieved" << std::endl;
		if (msg != NULL){
			// CONVERT ROS MSG TO CV MAT
			std::cout << "Converting Msg" << std::endl;
			image = cv_bridge::toCvShare(msg, "bgr8")->image;
			//~ image = cv_bridge::toCvShare(msg, "32FC1")->image;
			
			// DISPLAY CV MAT
			cv::imshow("Image Viewer", image);
			if(cv::waitKey(30) >= 0){
				std::cout << "Key out" << std::endl;
			}
		}
	}
	
	bag.close();
	
	return 0;
}
