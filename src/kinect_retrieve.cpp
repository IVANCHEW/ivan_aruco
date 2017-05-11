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
// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <ros/callback_queue.h>
#include <yaml-cpp/yaml.h>

#include "data_manager.cpp"

DataManagement dm;

int image_buffer_  = 0;
bool got_cloud_ = false;
bool got_image_ = false;
std::string package_path_;
pcl::visualization::PCLVisualizer viewer("Kinect Viewer");

// Retrieve msg from /kinect2/sd/points and converts to PointType variable
// before loading it into the Data Manager.
void cloud_callback(const sensor_msgs::PointCloud2& cloud_msg){
	got_cloud_ = true;
  // Convert from msg to pcl 
	pcl::PCLPointCloud2 pcl_pc;
	pcl_conversions::toPCL(cloud_msg,pcl_pc);
	pcl::PointCloud<PointType>::Ptr scene_cloud(new pcl::PointCloud<PointType>);
	pcl::fromPCLPointCloud2(pcl_pc,*scene_cloud);
	std::cout << "Cloud size: " << scene_cloud->points.size() << std::endl;
	dm.loadCloud(scene_cloud);
}

// Retrieves msg from /kinect2/sd/image_color_rect, converts to image, detects markers
// and undistorts the image before loading the image, rvec and tvec into the Data Manager.
void image_callback(const sensor_msgs::ImageConstPtr& msg){
	//~ std::cout << "Image callback, buffer: " << image_buffer_ << std::endl;
	if(image_buffer_>=30){
		got_image_ = true;
		try{
			cv::Mat image;
			cv::Mat unDistort;
			cv::Mat rvec;
			cv::Mat tvec;
			bool marker_found;
			image = cv_bridge::toCvShare(msg, "bgr8")->image;
			dm.loadFrame(image);
			std::cout << "Image Frame loaded" << std::endl;
		}
		catch(cv_bridge::Exception& e){
			ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		}
	}
	else{
		image_buffer_++;
	}
}

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI IntensityType;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::PointNormal NormalPointType;

int main (int argc, char** argv){
	std::cout << std::endl << "Kinect Retrieve Package" << std::endl;
  ros::init(argc, argv, "kinect_retrieve");
    
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  std::string save_path = package_path_ + "/kinect_retrieve_saves/";
  std::string point_cloud_topic_name = "/kinect2/sd/points";
  std::string image_topic_name = "/kinect2/sd/image_color_rect";
  
	// LOAD SINGLE IMAGE AND POINT CLOUD
	ros::NodeHandle nh_, nh_private_;
	double begin =ros::Time::now().toSec();
	double current = ros::Time::now().toSec();
	double time_limit = 5.0;
	
	std::cout << std::endl << "Subscribing to Kinect Image Topic..." << std::endl;
	image_transport::ImageTransport it(nh_);
	image_transport::Subscriber image_sub_ = it.subscribe(image_topic_name, 1, image_callback);
	while (current-begin < time_limit && !got_image_) {
		current =ros::Time::now().toSec();
		ros::spinOnce();
	}
	image_sub_.shutdown();
	
	begin =ros::Time::now().toSec();
	current = ros::Time::now().toSec();

	std::cout << std::endl << "Subscribing to Kinect Point Cloud Topic..." << std::endl;
	ros::Subscriber point_cloud_sub_;
	point_cloud_sub_ = nh_.subscribe(point_cloud_topic_name, 1, cloud_callback);
	while (current-begin < time_limit && !got_cloud_) {
		current =ros::Time::now().toSec();
		ros::spinOnce();
	}
	point_cloud_sub_.shutdown();
	
	bool retrieve_cloud_ = false;
	pcl::PointCloud<PointType>::Ptr cloud_a;
	retrieve_cloud_ = dm.getCloud(cloud_a);
	
	if (retrieve_cloud_){
		viewer.addPointCloud(cloud_a, "cloud_a");
		pcl::PCDWriter writer;
		writer.write<PointType> (save_path + "cloud.pcd", *cloud_a, false);
		std::cout << "Point Cloud saved" << std::endl;
	}else{
		std::cout << "No point cloud obtained" << std::endl;
	}
	
  cv::Mat image_display_;
  bool retrieve_image_ = false;
  retrieve_image_ = dm.getRawFrame(image_display_);
	
	if (retrieve_image_){
		cv::imwrite( save_path + "image.png", image_display_);
		std::cout << "Image Saved" << std::endl;
		
		std::cout << std::endl << "Starting Image Viewer..." << std::endl;
		cv::imshow("Image Viewer", image_display_);
		if(cv::waitKey(0) >= 0){
			std::cout << "Key out" << std::endl;
		}

	}else{
		std::cout << "No Image obtained" << std::endl;
	}
	
	if(retrieve_cloud_){
		std::cout << std::endl << "Starting Point Cloud Viewer..." << std::endl;
		while (!viewer.wasStopped ()){
			viewer.spinOnce();
		}
	}
	
	ros::shutdown();	
}
