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

// For Software control
int input_value_ = 0;
bool debug_;
bool got_image_ = false;
bool got_cloud_ = false;
int image_buffer_ = 0;

class DataManagement
{
	private:
	
		cv::Mat tvec;
		cv::Mat rvec;
		cv::Mat frame;
		std::vector <int> point_id;
		std::vector <int> cloud_index;
		std::vector < std::vector < float > > feature_desc;
		std::vector < std::vector < int > > feature_desc_index;
		int index_count = 0;
		int frame_height, frame_width;
		bool transformation_ready = false;
		bool image_ready = false;
		bool cloud_ready = false;
		bool pixel_point_ready = false;
		bool reading_pixel_point = false;
		bool parameters_ready = false;
		pcl::PointCloud<PointType>::Ptr cloud;
		
	public:
	
		void setParameters(double h, double w);
		void setPixelPointReady();
	
		void loadTransform(cv::Mat t, cv::Mat r);
		void loadFrame(cv::Mat f);
		void loadCloud(pcl::PointCloud<PointType>::Ptr &c);
		void loadPixelPoint(cv::Point2f p, int id);
		
		void getTransform(cv::Mat& t, cv::Mat& r);
		bool getFrame(cv::Mat& f);
		bool getCloud(pcl::PointCloud<PointType>::Ptr &r);
		bool getPixelPoint(cv::Point2f &p);
		bool getPointCloudIndex(int &index, int n);
		bool getPointIndexSize(int &n);
		
		void computeDescriptors();
		
		void arrangeDescriptors();
		
		void clearPixelPoints();
		
		bool statusTransform();
		
		void printDescriptors();
};

void DataManagement::setParameters(double h, double w){
	frame_height = (int) h;
	frame_width = (int) w;
	
	std::cout << std::endl << "Parameters set. W = " << frame_width << " H = " << frame_height << std::endl;
	
	parameters_ready = true;
}

void DataManagement::setPixelPointReady(){
		pixel_point_ready = true;
}

void DataManagement::loadTransform(cv::Mat t, cv::Mat r){
	tvec = t;
	rvec = r;
	transformation_ready = true;
}

void DataManagement::loadFrame(cv::Mat f){
		frame = f.clone();
		image_ready = true;
}

void DataManagement::loadCloud(pcl::PointCloud<PointType>::Ptr &c){
	if (!cloud_ready){
		cloud = c;
		cloud_ready = true;
	}
}

// Also computes corresponding cloud index with the given pixel position
void DataManagement::loadPixelPoint(cv::Point2f p, int id){
	std::cout << "Loading pixel point " << index_count << " . x: " << p.x << ", y: " << p.y << std::endl;
	int index_temp = p.y * frame_width + p.x;
	cloud_index.push_back(index_temp);
	point_id.push_back(id);
	pixel_point_ready = true;
	index_count++;
}

void DataManagement::getTransform(cv::Mat& t, cv::Mat& r){
	t = tvec;
	r = rvec;
}

bool DataManagement::getFrame(cv::Mat& f){
	if (image_ready){
		f = frame.clone();
		image_ready = false;
		return true;
	}
	else
		return false;
}

bool DataManagement::getCloud(pcl::PointCloud<PointType>::Ptr &r){
	if (cloud_ready){
		r = cloud;
		cloud_ready = false;
		return true;
	}
	else{
		return false;
	}
}

// Not used
bool DataManagement::getPixelPoint(cv::Point2f &p){
	if (pixel_point_ready){
		//~ p = pixel_point;
		pixel_point_ready = false;
		return true;
	}
	else{
		return false;
	}
}

bool DataManagement::getPointCloudIndex(int &index, int n){
	if (parameters_ready && pixel_point_ready){
		index = cloud_index[n];
		std::cout << "Point cloud index returned: " << index << std::endl;
		return true;
	}
	else{
		return false;
	}
}

bool DataManagement::getPointIndexSize(int &n){
		n = point_id.size();
		std::cout << "Point index size returned: " << n << std::endl;
		if (n>0){
			return true;
		}
		else{
			return false;
		}
}

void DataManagement::computeDescriptors(){
	if(pixel_point_ready && cloud_ready){
		int n = point_id.size();
		std::cout << "Computing descriptor, n: " << n << std::endl;
		for (int i = 0 ; i < n ; i++){
			std::vector<float> temp_desc;
			std::vector<int> temp_index;
			for (int j = 0 ; j < n ; j++){
				if(i!=j){
					float delta_x = cloud->points[cloud_index[i]].x - cloud->points[cloud_index[j]].x;
					float delta_y = cloud->points[cloud_index[i]].y - cloud->points[cloud_index[j]].y;
					float delta_z = cloud->points[cloud_index[i]].z - cloud->points[cloud_index[j]].z;
					float s = sqrt(pow(delta_x,2) + pow(delta_y,2) + pow(delta_z,2));
					temp_desc.push_back(s);
					temp_index.push_back(point_id[j]);
				}
			}
			feature_desc_index.push_back(temp_index);
			feature_desc.push_back(temp_desc);
		}
		
		printDescriptors();
		std::cout << "Finished computing descriptors... " << std::endl;
	}
	else if(!pixel_point_ready){
		std::cout << "Pixel points not ready, cannot compute descriptor" << std::endl;
	}
	else if(!cloud_ready){
		std::cout << "Point cloud not ready, cannot compute descriptor" << std::endl;
	}
}

void DataManagement::arrangeDescriptors(){
	
	std::cout << std::endl << "Sorting descriptors according to magnitude..." << std::endl;
	for (int n=0; n < feature_desc.size() ; n++){
		bool lowest_score = true;
		int front_of_index;
		std::vector <float> sorted_desc;
		std::vector <int> sorted_index;
		sorted_desc.push_back(feature_desc[n][0]);
		sorted_index.push_back(feature_desc_index[n][0]);
		
		for (int i=1; i<feature_desc[n].size(); i++){    
			lowest_score = true;  
			for (int  j=0 ; j < sorted_desc.size() ; j++){   
				if(feature_desc[n][i] <= sorted_desc[j]){
					front_of_index = j;
					lowest_score = false;
					break;
				}       
			}
			if (lowest_score==false){
				sorted_desc.insert(sorted_desc.begin() + front_of_index, feature_desc[n][i]);     
				sorted_index.insert(sorted_index.begin() + front_of_index, feature_desc_index[n][i]);           
			}
			else if(lowest_score==true){
				sorted_desc.push_back(feature_desc[n][i]);
				sorted_index.push_back(feature_desc_index[n][i]);
			}
		}
		
		feature_desc_index[n].swap(sorted_index);
		feature_desc[n].swap(sorted_desc);
	}
	
	printDescriptors();
	std::cout << "Descriptors Sorted..." << std::endl;
}

void DataManagement::clearPixelPoints(){
	std::vector <int> ().swap(point_id);
	std::vector <int> ().swap(cloud_index);
	index_count = 0;
	pixel_point_ready = false;
}

bool DataManagement::statusTransform(){
	return transformation_ready;
}

void DataManagement::printDescriptors(){
	for( int i = 0; i < point_id.size() ; i++){
		std::cout << " Descriptor (" << point_id[i] << ") : ";
		for (int j = 0 ; j < feature_desc[i].size() ; j++){
			std::cout << feature_desc[i][j] << "(" << feature_desc_index[i][j] << "), ";
		}
		std::cout << std::endl;
	}
}

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
		//~ std::cout << "No markers detected" << std::endl;
	}
	
	return marker_found;
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
			
			marker_found = false;
			marker_found = arucoPoseEstimation(image, target_id, tvec, rvec, camera_matrix, dist_coeffs, true);
			
			cv::undistort(image, unDistort, camera_matrix, dist_coeffs);
			std::cout << "Image Size: " << unDistort.size() << std::endl;
			dm.loadFrame(unDistort);
		}
		catch(cv_bridge::Exception& e){
			ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		}
	}
	else{
		image_buffer_++;
	}
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
  
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  std::string camera_name = "kinect_sd";
  std::string point_cloud_topic_name = "/kinect2/sd/points";
  std::string image_topic_name = "/kinect2/sd/image_color_rect";
  
  // CAMERA CALIBRATION
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  loadCalibrationMatrix(camera_name);
	focal_length = camera_matrix.at<double>(0,0);
	
	dm.setParameters(2*camera_matrix.at<double>(1,2), 2*camera_matrix.at<double>(0,2));
	
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
	ros::NodeHandle nh_, nh_private_;
	
	std::cout << std::endl << "Subscribing to Kinect Image Topic..." << std::endl;
	image_transport::ImageTransport it(nh_);
  image_transport::Subscriber image_sub_ = it.subscribe(image_topic_name, 1, image_callback);
	while (!got_image_) {
		ros::spinOnce();
	}
	image_sub_.shutdown();
	
	std::cout << std::endl << "Subscribing to Kinect Point Cloud Topic..." << std::endl;
  ros::Subscriber point_cloud_sub_;
  point_cloud_sub_ = nh_.subscribe(point_cloud_topic_name, 1, cloud_callback);
  while (!got_cloud_) {
		ros::spinOnce();
	}
	point_cloud_sub_.shutdown();
	
	//COMPUTE DESCRIPTORS
	dm.computeDescriptors();
	dm.arrangeDescriptors();
	
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

