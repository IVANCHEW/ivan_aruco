// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <ros/callback_queue.h>
#include <yaml-cpp/yaml.h>
#include <rosbag/bag.h>

#include "data_manager.cpp"

// POINT CLOUD PROCESSING
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>

std::string package_path_;
pcl::visualization::PCLVisualizer viewer("Kinect");
pcl::PointCloud<PointType>::Ptr cloud_a(new pcl::PointCloud<PointType>);

int count, input_value_;
bool debug_;
bool updateCloud = false;
bool active_threads_ = true;

void transformPC(pcl::PointCloud<PointType>::Ptr& source_cloud, float x, float y, float z, float deg, char axis)
{
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	// Linear Transformation
	transform_2.translation() << x, y, z;
	//Angular Transformation
	switch(axis){
		case 'x' :
			//~ cout << "X-axis rotation" << endl;
			transform_2.rotate (Eigen::AngleAxisf (deg, Eigen::Vector3f::UnitX()));
			break;
		case 'y' :
			//~ cout << "Y-axis rotation" << endl;
			transform_2.rotate (Eigen::AngleAxisf (deg, Eigen::Vector3f::UnitY()));
			break;
		case 'z' :
			//~ cout << "Z-axis rotation" << endl;
			transform_2.rotate (Eigen::AngleAxisf (deg, Eigen::Vector3f::UnitZ()));
			break;

		default :
			cout << "Invalid axis" << endl;
	}
	pcl::transformPointCloud (*source_cloud, *source_cloud, transform_2);
}

// PCL VIEWER KEYBOARD CALL BACK
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void){
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  std::string key_value = event.getKeySym();
  std::cout << key_value << std::endl;
  
  if (event.getKeySym () == "d" && event.keyDown ()){
    input_value_ = 1;
  }
}

int main (int argc, char** argv){
	std::cout << std::endl << "Test Bag Generator Package" << std::endl;
  ros::init(argc, argv, "test_bag_generator");
	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_("~");
	
	// Variable Initialisation
  package_path_ = ros::package::getPath("ivan_aruco");  
  std::string save_path_ = package_path_ + "/kinect_record_saves/test_cloud.bag"; 
  count = 0;
  
  // ROS Bag Initialisation
  rosbag::Bag bag;
	bag.open(save_path_, rosbag::bagmode::Write);
	
  // For debugging
  std::cout << "Package Path: " << package_path_ << std::endl;
  
	// Visualiser Test
	std::cout << "Adding test cloud" << std::endl;
	pcl::PCDReader reader;
	std::cout << "Reading point cloud" << std::endl;
	reader.read (package_path_ + "/test_frames/back.pcd", *cloud_a);		
	std::cout << "Point cloud size: " << cloud_a->points.size() << std::endl;
	viewer.addPointCloud (cloud_a,"cloud a");
  viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
  
	// START VIEWER
	float increment = 0.0005;
	while (active_threads_){
		transformPC(cloud_a, increment, 0, 0, 0, 'x');
		viewer.updatePointCloud(cloud_a, "cloud a");
		
		viewer.spinOnce();
		
		// CONVERSION TO ROS MSG FOR SAVE
		sensor_msgs::PointCloud2 cloud_msg_ros;
		pcl::toROSMsg(*cloud_a, cloud_msg_ros);
		
		// WRITING TO ROSBAG
		bag.write("cloud", ros::Time::now(), cloud_msg_ros);
		
		if(input_value_==1){
			active_threads_=false;
		}
	}
	
	bag.close();
	
  std::cout << std::endl << "End of script" << std::endl;
  
  return 0;
}
