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

// FOR POINT CLOUD 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI IntensityType;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::PointNormal NormalPointType;

std::string package_path_;
std::string folder_path_;
std::string bag_name_;
bool single_frame_, viewer_off_;
bool exit_replay_ = false;
double delay_time_;
pcl::visualization::PCLVisualizer viewer("Kinect Viewer");

int main (int argc, char** argv){

	std::cout << std::endl << "Rosbag Replay Started" << std::endl;
  ros::init(argc, argv, "rosbag_replay");
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_("~");
  
  ros::Publisher image_pub_ = nh_.advertise<sensor_msgs::Image>("/rosbag_replay/image", 1);
  ros::Publisher cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/rosbag_replay/cloud", 1);
  
  bool first_run_ = true;
  
  nh_private_.getParam("bag_name_", bag_name_);
  nh_private_.getParam("folder_name_", folder_path_);
  nh_private_.getParam("single_frame_", single_frame_);
  nh_private_.getParam("viewer_off_", viewer_off_);
  nh_private_.getParam("delay_time_", delay_time_);
  
  if (single_frame_){
		std::cout << "Single Frame Mode" << std::endl;
	}else{
		std::cout << "Continuous Mode" << std::endl;
	}
	
	if (viewer_off_){
		viewer.close();
	}
	
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  
  // VISUALISATION FOR DEBUGGING
  cv::namedWindow("Image Viewer",1);
  
	rosbag::Bag bag;
	bag.open(package_path_ + folder_path_ + bag_name_, rosbag::bagmode::Read);
	
	std::vector<std::string> topics;
	topics.push_back(std::string("image"));
	topics.push_back(std::string("cloud"));
	
	rosbag::View view(bag, rosbag::TopicQuery(topics));
	BOOST_FOREACH(rosbag::MessageInstance const m, view)
	{
		//std::cout << "Accessing Message" << std::endl;
		double current_time_;
		double last_time_ = ros::Time::now().toSec();
		bool delay = true;
		while(delay){
			current_time_ = ros::Time::now().toSec();
			if (current_time_ - last_time_ > delay_time_) delay = false;
		}
		
		sensor_msgs::Image::ConstPtr msg = m.instantiate<sensor_msgs::Image>();
		//std::cout << "Image message component retrieved" << std::endl;
		if (msg != NULL){
			// CONVERT ROS MSG TO CV MAT
			//std::cout << "Converting Image Msg" << std::endl;
			cv::Mat image;
			image = cv_bridge::toCvShare(msg, "bgr8")->image;
			
			// PUBLISH MESSAGE
			image_pub_.publish(msg);
			ros::spinOnce();
			
			// DISPLAY CV MAT
			if(!viewer_off_){
				if(single_frame_){
					cv::imshow("Image Viewer", image);
					if(cv::waitKey(0) >= 0){
						std::cout << "Next Frame" << std::endl;
					}
				}
				else{
					cv::imshow("Image Viewer", image);
					if(cv::waitKey(30) >= 0){
						std::cout << "Key out" << std::endl;
						exit_replay_ = true;
					}
				}
			}
			
			continue;
		}
		
		sensor_msgs::PointCloud2::ConstPtr cloud_from_bag = m.instantiate<sensor_msgs::PointCloud2>();
		//std::cout << "Cloud message component retrieved" << std::endl;
		
		if (cloud_from_bag != NULL)
		{
			// CONVERT ROS MSG TO PCL TYPE
			//std::cout << "Converting Cloud Msg" << std::endl;
			pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());
			pcl::fromROSMsg(*cloud_from_bag,*cloud);
			
			// PUBLISH MESSAGE
			cloud_pub_.publish(cloud_from_bag);
			ros::spinOnce();
			
			// DISPLAY POINT CLOUD
			if(!viewer_off_){
				if (first_run_){
					viewer.addPointCloud(cloud, "cloud_a");
					first_run_ = false;
				}else{
					viewer.updatePointCloud(cloud, "cloud_a");
				}
				viewer.spinOnce();
			}
			
		}
		
		if(exit_replay_)
			break;
		
	}
	
	std::cout << "End of replay" << std::endl;
	
	bag.close();
	
	return 0;
}
