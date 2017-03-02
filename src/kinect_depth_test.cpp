#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/console/print.h>
#include <pcl/correspondence.h>
#include <pcl/features/board.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/hv/hv_go.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkPLYReader.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sensor_msgs/PointCloud2.h>
#include <pthread.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI IntensityType;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::PointNormal NormalPointType;

std::string package_path_;
pcl::visualization::PCLVisualizer viewer("Kinect");
pcl::PointCloud<PointType>::Ptr cloud_a(new pcl::PointCloud<PointType>);

int count;
bool debug_;
bool updateCloud = false;

void transformPC(pcl::PointCloud<PointType>::Ptr& source_cloud, float x, float y, float z, float deg, char axis){
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	// Linear Transformation
	transform_2.translation() << x, y, z;
	//Angular Transformation
	switch(axis){
		case 'x' :
			transform_2.rotate (Eigen::AngleAxisf (deg, Eigen::Vector3f::UnitX()));
			break;
		case 'y' :
			transform_2.rotate (Eigen::AngleAxisf (deg, Eigen::Vector3f::UnitY()));
			break;
		case 'z' :
			transform_2.rotate (Eigen::AngleAxisf (deg, Eigen::Vector3f::UnitZ()));
			break;

		default :
			cout << "Invalid axis" << endl;
	}
	pcl::transformPointCloud (*source_cloud, *source_cloud, transform_2);
}

void cloud_callback(const sensor_msgs::PointCloud2& cloud_msg){
	// Convert from msg to pcl 
	pcl::PCLPointCloud2 pcl_pc;
	pcl_conversions::toPCL(cloud_msg,pcl_pc);
	//~ pcl::PointCloud<PointType>::Ptr scene_cloud(new pcl::PointCloud<PointType>);
	//~ pcl::fromPCLPointCloud2(pcl_pc,*scene_cloud);
	pcl::fromPCLPointCloud2(pcl_pc,*cloud_a);
	// Update viewer
	if (updateCloud){
		//~ viewer.updatePointCloud(scene_cloud, "cloud_1");
		viewer.updatePointCloud(cloud_a, "cloud a");
		std::cout << "Frame: " << count << std::endl;
		count++;
	}
	else{
		viewer.addPointCloud (cloud_a,"cloud a");
		updateCloud = false;
	}
}

// THREAD FUNCTIONS
void *start_viewer(void *threadid){
	long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting Point Cloud Viewer, Thread ID : " << tid << std::endl;
	viewer.spin();
	pthread_exit(NULL);
}

void *start_ros_callback(void *threadid){
	long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting ROS Callback Listener, Thread ID : " << tid << std::endl;
  std::cout << "Subscribing to Kinect Point Cloud Topic" << std::endl;
  ros::NodeHandle nh_, nh_private_;
	ros::Subscriber point_cloud_sub_;
	point_cloud_sub_ = nh_.subscribe("/kinect2/sd/points", 1, cloud_callback);
  ros::spin();
  ros::shutdown();
	pthread_exit(NULL);
}

void *start_test_thread(void *threadid){
	std::cout << "Test thread start" << std::endl;
	float increment = 0.0005;
	while (true){
		transformPC(cloud_a, increment, 0, 0, 0, 'x');
		viewer.updatePointCloud(cloud_a, "cloud a");
	}
}

int main (int argc, char** argv){
	std::cout << std::endl << "Kinect Depth Test Package" << std::endl;
  ros::init(argc, argv, "kinect_depth_test");
  
	// Variable Initialisation
  package_path_ = ros::package::getPath("ivan_aruco");   
  count = 0;
  
  // For debugging
  std::cout << "Package Path: " << package_path_ << std::endl;
  
	// START VIEWER THREAD
	pthread_t thread[2];
	int threadError;
	int i=0;
	
	threadError = pthread_create(&thread[i], NULL, start_viewer, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}

	// START ROS CALLBACK
	i++;
	threadError = pthread_create(&thread[i], NULL, start_ros_callback, (void *)i);
	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}
	
	pthread_exit(NULL);
	
  std::cout << std::endl << "End of script" << std::endl;
  
  return 0;
}
