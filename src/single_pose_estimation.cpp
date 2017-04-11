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

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI IntensityType;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::PointNormal NormalPointType;

// For ROS .yaml calibration
std::string yaml_path_;
std::string package_path_;

// For PCL visualisation
pcl::visualization::PCLVisualizer viewer("Kinect Viewer");

// For 2D camera parameters
cv::Mat camera_matrix;
cv::Mat dist_coeffs;
float focal_length;
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
float aruco_size = 0.08/2;
int target_id = 1;
std::string point_cloud_topic_name;
std::string image_topic_name;
 
// For SD resolution
std::string camera_name = "kinect_sd_single";

// For HD resolution
//~ std::string camera_name = "kinect";

// For Image Processing
int blur_param_ = 5;
int hsv_target_ = 145;
int hsv_threshold_ = 10;
int contour_area_min_ = 500;
int contour_area_max_ = 4000;
double contour_ratio_min_ = 6;
double contour_ratio_max_ = 9;

// For Software control
int input_value_ = 0;
bool debug_;
bool got_image_ = false;
bool got_cloud_ = false;
int image_buffer_ = 0;
bool aruco_detection_ = false;

#include "data_manager.cpp"
DataManagement dm;

// ARUCO Marker Detection function
bool arucoPoseEstimation(cv::Mat& input_image, int id, cv::Mat& tvec, cv::Mat& rvec, cv::Mat& mtx, cv::Mat& dist, bool draw_axis){
	// Contextual Parameters
	std::cout << std::endl << "Pose estimation called..." << std::endl;
	float aruco_square_size = aruco_size*2;
	bool marker_found = false;
	std::vector< int > marker_ids;
	std::vector< std::vector<cv::Point2f> > marker_corners, rejected_candidates;
	cv::Mat gray;
	
	cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
	cv::aruco::detectMarkers(gray, dictionary, marker_corners, marker_ids);	
	dm.clearPixelPoints();
	std::cout << "Number of markers detected: " << marker_ids.size() << std::endl;
	if (marker_ids.size() > 0){
		for (int i = 0 ; i < marker_ids.size() ; i++){
			std::cout << "Marker ID found: " << marker_ids[i] << std::endl;
			
			std::vector< std::vector<cv::Point2f> > single_corner(1);
			single_corner[0] = marker_corners[i];
			
			dm.loadPixelPoint(marker_corners[i][0], marker_ids[i]);
			
			cv::aruco::estimatePoseSingleMarkers(single_corner, aruco_square_size, mtx, dist, rvec, tvec);
			if (draw_axis){
				cv::aruco::drawDetectedMarkers(input_image, marker_corners, marker_ids);
				cv::aruco::drawAxis(input_image, mtx, dist, rvec, tvec, aruco_square_size/2);
			}
		}
		dm.setPixelPointReady();
		marker_found = true;
	}
	else{
		std::cout << "No markers detected" << std::endl;
	}
	
	return marker_found;
}

// Circular Marker Detection Function
bool circleEstimation (cv::Mat& input_image, int blur_param_, int hsv_target_, int hsv_threshold_ , int contour_area_min_, int contour_area_max_ ){
	
	int marker_count_ = 0;
	
	//#1 Get Image Shape Parameters
	//std::cout << "Step 1: Getting Image Shape Parameters" << std::endl;
	int row = input_image.rows;
	int col = input_image.cols;
	cv::imwrite(package_path_ + "/pose_estimation_frames/original_image.png", input_image);
	
	//#2 Median Blur Image
	cv::Mat image_blurred;
	cv::medianBlur(input_image, image_blurred, blur_param_);
	cv::imwrite(package_path_ + "/pose_estimation_frames/blurred_image.png", image_blurred);
	
	//#3 Apply HSV Filtering
	cv::Mat image_hsv;
	cv::Mat image_hsv_filtered;
	cv::cvtColor(image_blurred, image_hsv, CV_BGR2HSV);
	cv::inRange(image_hsv,cv::Scalar(hsv_target_ - hsv_threshold_,0,0), cv::Scalar(hsv_target_ + hsv_threshold_,255,255),image_hsv_filtered);
	cv::imwrite(package_path_ + "/pose_estimation_frames/hsv_image.png", image_hsv);
	cv::imwrite(package_path_ + "/pose_estimation_frames/hsv_filtered_image.png", image_hsv_filtered);
	
	//#4 Find Contours
	std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(image_hsv_filtered, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
  
	//#5 Filter Unnecessary Contours
	cv::Mat image_contour_filtered =  cv::Mat::zeros( input_image.size(), CV_8U);
	for (int i = 0 ; i < contours.size() ; i++){
			double contour_area = cv::contourArea(contours[i]);
			if((contour_area < contour_area_max_) && (contour_area > contour_area_min_)){
				
				//#6 Check for Child Contours
				bool marker_confirmed_ = false;
				for (int j = i; j < hierarchy.size() ; j++){
					if((hierarchy[j][3]==i) && (hierarchy[j][2]==-1)){
						double child_area_ = cv::contourArea(contours[j]);
						double contour_ratio_ = contour_area / child_area_;
						if((contour_ratio_max_ >= contour_ratio_) && (contour_ratio_min_ <= contour_ratio_)){
							marker_confirmed_ = true;
						}
					}
				}
				
				//#6 Compute Centroid and give temporary ID
				if(marker_confirmed_){
					std::cout << "Id: " << marker_count_ << ", area: " << contour_area << std::endl;
					std::vector<std::vector<cv::Point> > con = std::vector<std::vector<cv::Point> >(1, contours[i]);
					cv::Moments m = cv::moments(con[0], false);
					cv::Point2f p = cv::Point2f((int)(m.m10/m.m00) , (int)(m.m01/m.m00));
					cv::drawContours(input_image, con, -1, cv::Scalar(0, 255, 0), 1, 8);
					cv::circle(input_image, p, 1, cv::Scalar(0, 0, 255), 1, 8, 0);
					std::stringstream convert;
					convert << marker_count_;
					std::string s;
					s = convert.str();
					cv::putText(input_image, s, p, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8, false);
					dm.loadPixelPoint(p, marker_count_);
					marker_count_++;
				}
			}
	}
	cv::imwrite(package_path_ + "/pose_estimation_frames/contour_marked_image.png", input_image);
	
	//#7 Return True if sufficient markers are present to make pose estimate
	if (marker_count_ >= 3){
		dm.setPixelPointReady();
		return true;
	}else{
		return false;
	}
}

// CONFIGURATION AND SET-UP FUNCTIONS
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
	std::cout << "Updating Camera Parameters" << std::endl;
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
	if (config["image_topic"])
		image_topic_name = config["image_topic"].as<std::string>();
	if (config["cloud_topic"])
		point_cloud_topic_name = config["cloud_topic"].as<std::string>();
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
  ros::init(argc, argv, "kinect_pose_estimation");
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_("~");
  std::string image_path_, cloud_path_;
  
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
	nh_private_.getParam("image_path_", image_path_);
	nh_private_.getParam("cloud_path_", cloud_path_);
	nh_private_.getParam("blur_param_", blur_param_);
  nh_private_.getParam("hsv_target_", hsv_target_);
  nh_private_.getParam("hsv_threshold_", hsv_threshold_);
  nh_private_.getParam("contour_area_min_", contour_area_min_);
  nh_private_.getParam("contour_area_max_", contour_area_max_);
  nh_private_.getParam("contour_ratio_min_", contour_ratio_min_);
  nh_private_.getParam("contour_ratio_max_", contour_ratio_max_);
	nh_private_.getParam("aruco_detection_", aruco_detection_);
	
  // CAMERA CALIBRATION
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  loadCalibrationMatrix(camera_name);
	focal_length = camera_matrix.at<double>(0,0);
	
	dm.setParameters(2*camera_matrix.at<double>(1,2), 2*camera_matrix.at<double>(0,2),package_path_);
	
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
	
	// LOAD SINGLE IMAGE AND POINT CLOUD	
	double full_test_begin_ =ros::Time::now().toSec();
	double full_test_end_ = ros::Time::now().toSec();
	full_test_begin_ = ros::Time::now().toSec();
	
	//ADD IMAGE
	std::cout << "Adding Image" << std::endl;
	cv::Mat image;
	std::cout << "Reading image from: " << package_path_  + image_path_ << std::endl;
	image = cv::imread(package_path_ + image_path_, CV_LOAD_IMAGE_COLOR); 
	dm.loadFrame(image);
	
	//ADD POINT CLOUD
	std::cout << "Adding Point Cloud" << std::endl;
	pcl::PCDReader reader;
	std::cout << "Reading cloud from: " << package_path_  + cloud_path_ << std::endl;
	pcl::PointCloud<PointType>::Ptr cloud_load(new pcl::PointCloud<PointType>);
	reader.read (package_path_ + cloud_path_, *cloud_load);
	dm.loadCloud(cloud_load);		
	
	//COMPUTE DESCRIPTORS
	std::cout << "Begin Descriptor Computation" << std::endl;
	cv::Mat tvec;
	cv::Mat rvec;
	
	if(aruco_detection_){
		std::cout << "Using ARUCO Detection" << std::endl;
		arucoPoseEstimation(image, 0, tvec, rvec, camera_matrix, dist_coeffs, true);
	}else{
		std::cout << "Using Circular Marker Detection" << std::endl;
		dm.detectMarkers(blur_param_, hsv_target_, hsv_threshold_ , contour_area_min_, contour_area_max_, contour_ratio_min_, contour_ratio_max_, false);
		dm.computeDescriptors();
		dm.arrangeDescriptors();
	}
	
	full_test_end_ = ros::Time::now().toSec();
	std::cout << "Time taken for computation : " << full_test_end_  - full_test_begin_ << std::endl;
	
	// VIEW DETECTED POINTS ON THE CLOUD VIEWER
	// VIEWER PARAMETERS
	std::cout << std::endl << "Starting Point Cloud Viewer..." << std::endl;
	bool retrieve_cloud_ = false;
	bool retrieve_index_ = false;
	int highlight_size_ = 0;
	int cloud_index;
  pcl::PointCloud<PointType>::Ptr cloud_a;
  pcl::PointCloud<PointType>::Ptr	highlight_cloud (new pcl::PointCloud<PointType>);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> highlight_color_handler (highlight_cloud, 255, 0, 0);
  retrieve_cloud_ = dm.getCloud(cloud_a);
  
  if (retrieve_cloud_){
		viewer.addPointCloud(cloud_a, "cloud_a");
		retrieve_index_ = dm.getPointIndexSize(highlight_size_);
		std::cout << "Highlight points size: " << highlight_size_ << std::endl;
		if (retrieve_index_){
			for (int n=0 ; n < highlight_size_ ; n++){
				dm.getPointCloudIndex(cloud_index, n);
				highlight_cloud->points.push_back(cloud_a->points[cloud_index]);
			}
			viewer.addPointCloud (highlight_cloud, highlight_color_handler, "Highlight Cloud");
			viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "Highlight Cloud");
		}
	}

	// VIEW THE CORRESPONDING IMAGE
	std::cout << std::endl << "Starting Image Viewer..." << std::endl;
  cv::Mat image_display_;
  bool retrieve_image_ = false;
  retrieve_image_ = dm.getFrame(image_display_);
  if(retrieve_image_){
		cv::imshow("Image Viewer", image_display_);
		if(cv::waitKey(0) >= 0){
			std::cout << "Key out" << std::endl;
		}
	}
	
	while (!viewer.wasStopped ()){
		viewer.spinOnce();
	}
	
	ros::shutdown();
	
  return 0;
}

