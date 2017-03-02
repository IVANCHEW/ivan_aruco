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
int input_value_ = 0;
bool debug_;
bool updateCloud = false;
bool retrieve_cloud_ = false;
bool stop_all_ = false;

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
  retrieve_cloud_ = true;
}

// THREAD FUNCTIONS
void *start_viewer(void *threadid){
  long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting Point Cloud Viewer, Thread ID : " << tid << std::endl;
  while (!viewer.wasStopped ()){
    viewer.spinOnce();
    boost::this_thread::sleep (boost::posix_time::microseconds (100)); 
    if (input_value_ == 1){
      viewer.close();
    }
    input_value_ == 0;
  }
  stop_all_ = true;
  std::cout << "Exiting Viewer thread" << std::endl;
  pthread_exit(NULL);
}

void *start_test_thread(void *threadid){
	std::cout << "Test thread start" << std::endl;
	float increment = 0.0005;
	while (!stop_all_){
		transformPC(cloud_a, increment, 0, 0, 0, 'x');
		viewer.updatePointCloud(cloud_a, "cloud a");
	}
	std::cout << "Exiting Test Thread" << std::endl;
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
  
  // SINGLE CLOUD RETRIEVAL
  double begin =ros::Time::now().toSec();
  double current = ros::Time::now().toSec();
  retrieve_cloud_ = false;
  while (current-begin<5 && !retrieve_cloud_) {
    current =ros::Time::now().toSec();
    ros::spinOnce();
  }
  point_cloud_sub_.shutdown();
  ros::shutdown();
  
  // CONTINUOUS CLOUD RETRIEVAL
  //~ ros::spin();
  //~ ros::shutdown();
  
  std::cout << "Size of point cloud: " << cloud_a->points.size() << std::endl;
  std::cout << "Exiting ROS callback thread" << std::endl;
  pthread_exit(NULL);
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
  std::cout << std::endl << "Kinect Depth Test Package" << std::endl;
  ros::init(argc, argv, "kinect_depth_test");
  
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  count = 0;
  
  // DEBUGGING
  std::cout << "Package Path: " << package_path_ << std::endl;
  
  // THREADING
  pthread_t thread[2];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  int thread_error_;
  int i=0;
  void *status;
  
	// Visualiser Test
	std::cout << "Adding test cloud" << std::endl;
	pcl::PCDReader reader;
	std::cout << "Reading point cloud" << std::endl;
	reader.read (package_path_ + "/test_frames/back.pcd", *cloud_a);		
	std::cout << "Point cloud size: " << cloud_a->points.size() << std::endl;
	viewer.addPointCloud (cloud_a,"cloud a");
	
  // PREPARE VIEWER THREAD
  viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
  thread_error_ = pthread_create(&thread[i], NULL, start_viewer, (void *)i);
  if (thread_error_){
    std::cout << "Error:unable to create thread," << thread_error_ << std::endl;
    exit(-1);
  }

  // PREPARE ROS CALLBACK
  i++;
  //~ thread_error_ = pthread_create(&thread[i], NULL, start_ros_callback, (void *)i);
  thread_error_ = pthread_create(&thread[i], NULL, start_test_thread, (void *)i);
  if (thread_error_){
    std::cout << "Error:unable to create thread," << thread_error_ << std::endl;
    exit(-1);
  }
  
  // INITIATE THREADS
  pthread_attr_destroy(&attr);
  
  for(int j=0; j < 1; j++ ){
    thread_error_ = pthread_join(thread[i], &status);
    if (thread_error_){
      std::cout << "Error:unable to join," << thread_error_ << std::endl;
      exit(-1);
    }
  }
  
  std::cout << std::endl << "End of main script" << std::endl;
  return 0;
}
