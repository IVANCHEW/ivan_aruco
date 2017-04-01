// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <ros/callback_queue.h>
#include <yaml-cpp/yaml.h>

// POINT CLOUD PROCESSING
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZ PointType;
std::string package_path_;

pcl::visualization::PCLVisualizer viewer("Kinect Viewer");
pcl::PCDReader reader;

int main (int argc, char** argv){
  std::cout << std::endl << "Pose Estimation Cloud Viewer Package" << std::endl;
  ros::init(argc, argv, "pose_estimation_cloud_viewer");
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_("~");
  std::string folder_name_;
  
	nh_private_.getParam("folder_name", folder_name_);
	
	// VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  
  //~ std::string file_path = package_path_ + "/pose_estimation_frames/Test2/";
  //~ if (nh_.hasParam("folder_name"))
	//~ {
		//~ std::cout << "Parameter found" << std::endl;
	//~ }
  //~ nh_.getParam("folder_name", folder_name_);
  
  std::string file_path = package_path_ + folder_name_;
  std::cout << "Accessing cloud and highlights of: " << file_path << std::endl;
  
  pcl::PointCloud<PointType>::Ptr	cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr	highlights (new pcl::PointCloud<PointType>);
  reader.read(file_path + "cloud.pcd", *cloud);
  reader.read(file_path + "highlight_cloud.pcd", *highlights);

	viewer.addPointCloud(cloud, "cloud");
	pcl::visualization::PointCloudColorHandlerCustom<PointType> highlight_color_handler (highlights, 255, 0, 0);
	viewer.addPointCloud (highlights, highlight_color_handler, "Highlight Cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "Highlight Cloud");
	
	while (!viewer.wasStopped ()){
		viewer.spinOnce();
	}
	
	ros::shutdown();
	
  return 0;
  
}
