#include "data_manager.cpp"

DataManagement dm;

// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <ros/callback_queue.h>
#include <yaml-cpp/yaml.h>

// POINT CLOUD PROCESSING
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

// MISC
#include <pthread.h>

// IMAGE PROCESSING
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>

std::string cloud_topic_ = "/rosbag_replay/cloud";
std::string image_topic_ = "/rosbag_replay/image";
bool stop_all_ = false;
bool next_frame_ = true;
std::string package_path_;
std::string yaml_path_;

// CAMERA PARAMETERS
cv::Mat camera_matrix;
cv::Mat dist_coeffs;
float focal_length;

// For Image Processing
int blur_param_;
int hsv_target_;
int hsv_threshold_;
int contour_area_min_;
int contour_area_max_;
double contour_ratio_min_;
double contour_ratio_max_;

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners, std::string patternType){
    corners.clear();

    if (patternType == "CHESSBOARD" || patternType == "CIRCLES_GRID"){
			for( int i = 0; i < boardSize.height; ++i )
					for( int j = 0; j < boardSize.width; ++j )
							corners.push_back(cv::Point3f(float( j*squareSize ), float( i*squareSize ), 0));
		}

    else if (patternType == "ASYMMETRIC_CIRCLES_GRID"){
			for( int i = 0; i < boardSize.height; i++ )
					for( int j = 0; j < boardSize.width; j++ )
							corners.push_back(cv::Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
		}

}

void updateParameters(YAML::Node config){
	if (config["fx"])
    camera_matrix.at<double>(0,0) = config["fx"].as<double>();
	if (config["fx"])
    camera_matrix.at<double>(1,1) = config["fx"].as<double>();
	if (config["x0"])
    camera_matrix.at<double>(0,2) = config["x0"].as<double>();
	if (config["y0"])
    camera_matrix.at<double>(1,2) = config["y0"].as<double>();
  if (config["k1"])
    dist_coeffs.at<double>(0,0) = config["k1"].as<double>();
  if (config["k2"])
    dist_coeffs.at<double>(1,0) = config["k2"].as<double>();
  if (config["k3"])
    dist_coeffs.at<double>(4,0) = config["k3"].as<double>();
  if (config["p1"])
    dist_coeffs.at<double>(2,0) = config["p1"].as<double>();
  if (config["p2"])
    dist_coeffs.at<double>(3,0) = config["p2"].as<double>();
}

void loadCalibrationMatrix(std::string camera_name_){
	// Updates Parameter with .yaml file
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
	camera_matrix.at<double>(2,2) = 1;
  yaml_path_ = ros::package::getPath("ivan_aruco") + "/config/camera_info.yaml";
  YAML::Node config;
  try 
  {
    config = YAML::LoadFile(yaml_path_);
  } 
  catch (YAML::Exception &e) 
  {
    ROS_ERROR_STREAM("YAML Exception: " << e.what());
    exit(EXIT_FAILURE);
  }
  if (!config[camera_name_])
  {
    ROS_ERROR("Cannot find default parameters in yaml file: %s", yaml_path_.c_str());
    exit(EXIT_FAILURE);
  }
  updateParameters(config[camera_name_]);
  
}

void cloud_callback(const sensor_msgs::PointCloud2& cloud_msg){
  // Convert from msg to pcl 
  if (next_frame_){
		pcl::PCLPointCloud2 pcl_pc;
		pcl_conversions::toPCL(cloud_msg,pcl_pc);
		pcl::PointCloud<PointType>::Ptr scene_cloud(new pcl::PointCloud<PointType>);
		pcl::fromPCLPointCloud2(pcl_pc,*scene_cloud);
		dm.loadCloud(scene_cloud);
	}
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){
	if(next_frame_){
		try{
			cv::Mat image;
			bool marker_found;
			image = cv_bridge::toCvShare(msg, "bgr8")->image;
			dm.loadFrame(image);
		}
		catch(cv_bridge::Exception& e){
			ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		}
	}
}

void *start_ros_callback(void *threadid){
  long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting ROS Callback Listener, Thread ID : " << tid << std::endl;
  ros::NodeHandle nh_, nh_private_;
  ros::CallbackQueue cb_queue_;
	
  // SUBSCRIBE TO POINT CLOUD TOPIC
  std::cout << "Subscribing to Kinect Point Cloud Topic" << std::endl;
  ros::Subscriber point_cloud_sub_;
  point_cloud_sub_ = nh_.subscribe(cloud_topic_, 1, cloud_callback);
  
  //~ // SUBSCRIBE TO 2D IMAGE TOPIC
  std::cout << "Subscribing to Kinect Image Topic" << std::endl;
	image_transport::ImageTransport it(nh_);
  image_transport::Subscriber image_sub_ = it.subscribe(image_topic_, 1, image_callback);
  // CONTINUOUS RETRIEVAL
  while (!stop_all_){
		cb_queue_.callAvailable();
		ros::spinOnce();
	}
	
	image_sub_.shutdown();
	point_cloud_sub_.shutdown();
  ros::shutdown();
  
  std::cout << "Exiting ROS callback thread" << std::endl;
  pthread_exit(NULL);
}

void *start_image_viewer(void *threadid){
	long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting Image Viewer, Thread ID : " << tid << std::endl;
  
  // VIEWER PARAMETER
  cv::Mat image_display_;
  bool retrieve_image_ = false;
  double current_time_ = ros::Time::now().toSec();
  double last_retrieved_time_ = ros::Time::now().toSec();
  
  // VIEWER LOOP
  while (!stop_all_){
		current_time_ = ros::Time::now().toSec();
		
		// NODE TERMINATION CONDITION
		if(current_time_ - last_retrieved_time_ > 2.0){
			stop_all_  = true;
		}
		
		if(dm.getImageLoadStatus()){
			dm.detectMarkers(blur_param_, hsv_target_, hsv_threshold_ , contour_area_min_, contour_area_max_, contour_ratio_min_, contour_ratio_max_);
			retrieve_image_ = dm.getFrame(image_display_);
		}
		
		if(retrieve_image_){
			last_retrieved_time_ = ros::Time::now().toSec();
			cv::imshow("Image Viewer", image_display_);
			if(cv::waitKey(30) >= 0) {
				std::cout << "Key out" << std::endl;
				stop_all_ = true;
			}
		}
		retrieve_image_ = false;
	}
	std::cout << "Exiting Image Viewer Thread" << std::endl;
	pthread_exit(NULL);
}

int main (int argc, char** argv){
  std::cout << std::endl << "Replay Analysis Package" << std::endl;
  ros::init(argc, argv, "replay_analysis");
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_("~");
  
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  
  nh_private_.getParam("blur_param_", blur_param_);
  nh_private_.getParam("hsv_target_", hsv_target_);
  nh_private_.getParam("hsv_threshold_", hsv_threshold_);
  nh_private_.getParam("contour_area_min_", contour_area_min_);
  nh_private_.getParam("contour_area_max_", contour_area_max_);
  nh_private_.getParam("contour_ratio_min_", contour_ratio_min_);
  nh_private_.getParam("contour_ratio_max_", contour_ratio_max_);
  
  // CAMERA CALIBRATION
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  loadCalibrationMatrix("kinect_sd");
	focal_length = camera_matrix.at<double>(0,0);
	dm.setParameters(2*camera_matrix.at<double>(1,2), 2*camera_matrix.at<double>(0,2), package_path_);
	
  // DEBUGGING
  std::cout << "Package Path: " << package_path_ << std::endl;
	std::cout << std::endl << "Calibration Matrix: " << std::endl << std::setprecision(5);
	for (int i=0 ; i<3 ; i++){
		std::cout << "[ " ;
		for (int j=0 ; j<3 ; j++)
			std::cout << camera_matrix.at<double>(i,j) << " ";
		std::cout << "]" << std::endl;
	}
	std::cout << std::endl << "Focal Length: " << focal_length << std::endl;
	
	std:: cout << std::endl << "Distortion Matrix: " << std::endl << "[ ";
	for (int i=0 ; i<5 ; i++){
		std::cout << dist_coeffs.at<double>(i,0) << " ";
	}
	std::cout << "]" << std::endl;
	
  // THREADING
  pthread_t thread[3];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  int thread_error_;
  int i=0;
  void *status;
 
   // PREPARE ROS CALLBACK
  thread_error_ = pthread_create(&thread[i], NULL, start_ros_callback, (void *)i);
  if (thread_error_){
    std::cout << "Error:unable to create thread," << thread_error_ << std::endl;
    exit(-1);
  }
  
  // PREPARE VIEWER THREAD
  //~ i++;
  //~ viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
  //~ thread_error_ = pthread_create(&thread[i], NULL, start_cloud_viewer, (void *)i);
  //~ if (thread_error_){
    //~ std::cout << "Error:unable to create thread," << thread_error_ << std::endl;
    //~ exit(-1);
  //~ }

  // PREPARE IMAGE VIEWER CALLBACK
  i++;
  thread_error_ = pthread_create(&thread[i], NULL, start_image_viewer, (void *)i);
  if (thread_error_){
    std::cout << "Error:unable to create thread," << thread_error_ << std::endl;
    exit(-1);
  }
  
  // INITIATE THREADS
  pthread_attr_destroy(&attr);
  
  //initially j < 1
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
