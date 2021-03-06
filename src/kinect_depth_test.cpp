// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <ros/callback_queue.h>
#include <yaml-cpp/yaml.h>

// POSE ESTIMATION
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

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

// For ROS .yaml calibration
std::string yaml_path_;
std::string package_path_;

// For PCL visualisation
pcl::visualization::PCLVisualizer viewer("Kinect Viewer");

// For 2D camera parameters
cv::Mat camera_matrix;
cv::Mat dist_coeffs;
float focal_length;

// For Software control
int input_value_ = 0;
bool debug_;
bool stop_all_ = false;
bool next_frame_ = true;
bool next_cloud_ = true;

// For User Configuration
std::string cloud_topic_, image_topic_;
int blur_param_;
int hsv_target_;
int hsv_threshold_;
int contour_area_min_;
int contour_area_max_;
double contour_ratio_min_;
double contour_ratio_max_;
bool aruco_detection_;

// For Pose Estimation
int gc_threshold_;
float gc_size_;
float desc_match_thresh_;

#include "data_manager.cpp"

DataManagement dm;


void generateTestModelCloud(pcl::PointCloud<PointType>::Ptr &cloud){
	cloud->width = 6;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);
	
	for(int i=0; i<3; i++){
		cloud->points[i].x = 0;
		cloud->points[i].y = 0;
		cloud->points[i].z = 0;
	}
	
	cloud->points[1].x = 0.08;
	cloud->points[2].y = 0.2;
	cloud->points[3].z = 0.3;
	cloud->points[4].x = 0.1;
	cloud->points[4].y = 0.2;
	cloud->points[5].x = 0.4;
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

// Retrieve data from .yaml configuration file and load them into the
// global calibration and distortion matrix. 
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
		next_cloud_ = false;
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

void *start_viewer(void *threadid){
	long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting Image Viewer, Thread ID : " << tid << std::endl;
  
  // IMAGE VIEWER PARAMETER
  cv::Mat image_display_;
  cv::namedWindow("Image Viewer",1);
  bool retrieve_image_ = false;
  double current_time_ = ros::Time::now().toSec();
  double last_retrieved_time_ = ros::Time::now().toSec();
  
  // POINT CLOUD VIEWER PARAMETER
  pcl::visualization::PCLVisualizer viewer("Kinect Viewer");
  viewer.setCameraPosition(0.0308721, 0.0322514, -1.05573, 0.0785146, -0.996516, -0.0281465);
  viewer.setPosition(49, 540);
  pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr highlight_cloud (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_cloud_ (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr transformed_cloud_ (new pcl::PointCloud<PointType> ());
  pcl::visualization::PointCloudColorHandlerCustom<PointType> highlight_color_handler (highlight_cloud, 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler_ (model_cloud_, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> transformed_color_handler_ (transformed_cloud_, 0, 0, 255);
  
  bool first_cloud_ = true;
  bool retrieve_cloud_ = false;
  bool retrieve_index_ = false;
  
  //TO BE DELETED
  generateTestModelCloud(model_cloud_);
  dm.loadDatabaseDescriptors(model_cloud_);
  
  // VIEWER LOOP
  while (!stop_all_){
		current_time_ = ros::Time::now().toSec();
		
		// NODE TERMINATION CONDITION
		if(current_time_ - last_retrieved_time_ > 5.0){
			std::cout << "Terminating because of frame retrieval delay" << std::endl;
			stop_all_  = true;
		}
		
		// PROCESS IMAGE
		if(dm.getCloudAndImageLoadStatus()){
			next_frame_ = false;
			dm.detectMarkers(blur_param_, hsv_target_, hsv_threshold_ , contour_area_min_, contour_area_max_, contour_ratio_min_, contour_ratio_max_, aruco_detection_);
			dm.computeDescriptors();
			dm.getMatchingDescriptor();
			retrieve_image_ = dm.getFrame(image_display_);
			retrieve_cloud_ = dm.getCloud(cloud);
			next_frame_ = true;
		}
		
		// DISPLAY IMAGE
		if(retrieve_image_){
			last_retrieved_time_ = ros::Time::now().toSec();
			cv::imshow("Image Viewer", image_display_);
			if(cv::waitKey(30) >= 0) {
				std::cout << "Key out" << std::endl;
				stop_all_ = true;
			}
		}
		
		// DISPLAY CLOUD
		if (retrieve_cloud_){
			if (first_cloud_){
				std::cout << "Added new cloud to viewer" << std::endl;
				viewer.addPointCloud(cloud, "cloud");
				highlight_cloud->points.push_back(cloud->points[0]);
				viewer.addPointCloud (highlight_cloud, highlight_color_handler, "Highlight Cloud");
				viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "Highlight Cloud");
				transformed_cloud_->points.push_back(cloud->points[0]);
				viewer.addPointCloud(transformed_cloud_, transformed_color_handler_, "transformed");
				viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "transformed");
				first_cloud_ = false;
			}
			else{
				viewer.updatePointCloud(cloud, "cloud");
				if(dm.getHighlightCloud(highlight_cloud)){
					viewer.updatePointCloud(highlight_cloud, highlight_color_handler, "Highlight Cloud");
				}
			}
			
			Eigen::Matrix4f estimated_pose_;
			if(dm.computePoseEstimate(estimated_pose_, gc_size_, gc_threshold_)){
				ROS_DEBUG("Pose Found");
				pcl::transformPointCloud (*model_cloud_, *transformed_cloud_, estimated_pose_);
				viewer.updatePointCloud(transformed_cloud_, transformed_color_handler_, "transformed");
			}
			
			dm.clearPixelPoints();
			dm.clearDescriptors();
			viewer.spinOnce();
		}

		retrieve_cloud_ = false;
		retrieve_image_ = false;
		retrieve_index_ = false;
	}
	std::cout << "Exiting Image Viewer Thread" << std::endl;
	pthread_exit(NULL);
}

// Thread to retrieve ROS messages from topics subscribed
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
  nh_private_.getParam("aruco_detection_", aruco_detection_);
  nh_private_.getParam("image_topic_", image_topic_);
  nh_private_.getParam("cloud_topic_", cloud_topic_);
  nh_private_.getParam("desc_match_thresh_", desc_match_thresh_);
  dm.setDescMatchThreshold(desc_match_thresh_);

  nh_private_.getParam("gc_size_", gc_size_);
  nh_private_.getParam("gc_threshold_", gc_threshold_);
  nh_private_.getParam("debug_", debug_);
  if (debug_)
	{
		std::cout << "Debug Mode ON" << std::endl;
		pcl::console::setVerbosityLevel(pcl::console::L_INFO);
		ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
	}
	else
	{
		std::cout << "Debug Mode OFF" << std::endl;
		pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
		ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	}
	
  // CAMERA CALIBRATION
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  loadCalibrationMatrix("kinect_sd");
	focal_length = camera_matrix.at<double>(0,0);
	dm.setParameters(2*camera_matrix.at<double>(1,2), 2*camera_matrix.at<double>(0,2), package_path_);
	dm.setCameraParameters(camera_matrix, dist_coeffs);
	
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
  
  // PREPARE VIEWER THREAD
  viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
  thread_error_ = pthread_create(&thread[i], NULL, start_viewer, (void *)i);
  if (thread_error_){
    std::cout << "Error:unable to create thread," << thread_error_ << std::endl;
    exit(-1);
  }

  // PREPARE ROS CALLBACK
  i++;
  thread_error_ = pthread_create(&thread[i], NULL, start_ros_callback, (void *)i);
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
