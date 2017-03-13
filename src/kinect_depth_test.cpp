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
bool stop_all_ = false;
bool next_frame_ = true;
bool next_cloud_ = true;

class DataManagement
{
	private:
	
		cv::Mat tvec;
		cv::Mat rvec;
		cv::Mat frame;
		std::vector <int> point_index;
		std::vector <int> cloud_index;
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
	
		void loadTransform(cv::Mat t, cv::Mat r);
		void loadFrame(cv::Mat f);
		void loadCloud(pcl::PointCloud<PointType>::Ptr &c);
		void loadPixelPoint(cv::Point2f p);
		
		void getTransform(cv::Mat& t, cv::Mat& r);
		bool getFrame(cv::Mat& f);
		bool getCloud(pcl::PointCloud<PointType>::Ptr &r);
		bool getPixelPoint(cv::Point2f &p);
		bool getPointCloudIndex(int &index, int n);
		bool getPointIndexSize(int &n);
		void clearPixelPoints();
		
		bool statusTransform();
};

void DataManagement::setParameters(double h, double w){
	frame_height = (int) h;
	frame_width = (int) w;
	
	std::cout << std::endl << "Parameters set. W = " << frame_width << " H = " << frame_height << std::endl;
	
	parameters_ready = true;
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
void DataManagement::loadPixelPoint(cv::Point2f p){
	int index_temp = p.y * frame_width + p.x;
	cloud_index.push_back(index_temp);
	point_index.push_back(index_count);
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
		return true;
	}
	else{
		return false;
	}
}

bool DataManagement::getPointIndexSize(int &n){
		n = point_index.size();
		if (n>0){
			return true;
		}
		else{
			return false;
		}
}

void DataManagement::clearPixelPoints(){
	std::vector <int> ().swap(point_index);
	std::vector <int> ().swap(cloud_index);
}

bool DataManagement::statusTransform(){
	return transformation_ready;
}

DataManagement dm;

// POINT CLOUD MANIPULATION FUNCTIONS
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

// 2D IMAGE MANIPULATION FUNCTIONS

// Single Marker Detection function
bool arucoPoseEstimation(cv::Mat& input_image, int id, cv::Mat& tvec, cv::Mat& rvec, cv::Mat& mtx, cv::Mat& dist, bool draw_axis){
	// Contextual Parameters
	//~ std::cout << "Pose estimation called" << std::endl;
	float aruco_square_size = aruco_size*2;
	bool marker_found = false;
	std::vector< int > marker_ids;
	std::vector< std::vector<cv::Point2f> > marker_corners, rejected_candidates;
	cv::Mat gray;
	
	cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
	cv::aruco::detectMarkers(gray, dictionary, marker_corners, marker_ids);	
	dm.clearPixelPoints();
	if (marker_ids.size() > 0){
		for (int i = 0 ; i < marker_ids.size() ; i++){
			//~ std::cout << "Marker IDs found: " << marker_ids[i] << std::endl;
			
			std::vector< std::vector<cv::Point2f> > single_corner(1);
			single_corner[0] = marker_corners[i];
			
			dm.loadPixelPoint(marker_corners[i][0]);
			
			cv::aruco::estimatePoseSingleMarkers(single_corner, aruco_square_size, mtx, dist, rvec, tvec);
			if (draw_axis){
				cv::aruco::drawDetectedMarkers(input_image, marker_corners, marker_ids);
				cv::aruco::drawAxis(input_image, mtx, dist, rvec, tvec, aruco_square_size/2);
			}
			//~ std::cout << "Marker found : aruco pose estimation" << std::endl;
		}
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
  // Convert from msg to pcl 
  if (next_frame_){
		pcl::PCLPointCloud2 pcl_pc;
		pcl_conversions::toPCL(cloud_msg,pcl_pc);
		pcl::PointCloud<PointType>::Ptr scene_cloud(new pcl::PointCloud<PointType>);
		pcl::fromPCLPointCloud2(pcl_pc,*scene_cloud);
		//~ std::cout << "Cloud size: " << scene_cloud->points.size() << std::endl;
		dm.loadCloud(scene_cloud);
	}
}

// Retrieves msg from /kinect2/sd/image_color_rect, converts to image, detects markers
// and undistorts the image before loading the image, rvec and tvec into the Data Manager.
void image_callback(const sensor_msgs::ImageConstPtr& msg){
	std::cout << "Image callback" << std::endl;
	
	if(next_frame_){
		next_cloud_ = false;
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
			//~ std::cout << "Image Size: " << unDistort.size() << std::endl;
			dm.loadFrame(unDistort);
		}
		catch(cv_bridge::Exception& e){
			ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		}
	}
}

// THREAD FUNCTIONS

// Thread to load point clouds into the PCL visualiser
void *start_cloud_viewer(void *threadid){
  long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting Point Cloud Viewer, Thread ID : " << tid << std::endl;
  
  // VIEWER PARAMETERS
  pcl::PointCloud<PointType>::Ptr cloud_a;
  pcl::PointCloud<PointType>::Ptr	highlight_cloud (new pcl::PointCloud<PointType>);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> highlight_color_handler (highlight_cloud, 255, 100, 0);
  
  bool update_ = false;
  bool retrieve_cloud_ = false;
  
  // CLOUD ASSOCIATION TEST
  bool retrieve_index_ = false;
  int cloud_index;
  int highlight_size_;
  // VIEWER LOOP
  while (!viewer.wasStopped ()){
    retrieve_cloud_ = dm.getCloud(cloud_a);
        
    if (retrieve_cloud_){
			if (!update_){
				std::cout << "Added new cloud to viewer" << std::endl;
				viewer.addPointCloud(cloud_a, "cloud_a");
				highlight_cloud->points.push_back(cloud_a->points[0]);
				viewer.addPointCloud (highlight_cloud, highlight_color_handler, "Highlight Cloud");
				viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "Highlight Cloud");
				update_ = true;
			}
			else{
				viewer.updatePointCloud(cloud_a, "cloud_a");
				retrieve_index_ = dm.getPointIndexSize(highlight_size_);
				if (retrieve_index_){
					//Display distinguished key point
					highlight_cloud->points.clear();
					for (int n=0 ; n < highlight_size_ ; n++){
						dm.getPointCloudIndex(cloud_index, n);
						highlight_cloud->points.push_back(cloud_a->points[cloud_index]);
					}
					viewer.updatePointCloud(highlight_cloud, highlight_color_handler, "Highlight Cloud");
				}
			}
		}

		// FROM KEYBOARD INPUT
    if (input_value_ == 1){
      viewer.close();
    }
    input_value_ == 0;
    viewer.spinOnce();
    boost::this_thread::sleep (boost::posix_time::microseconds (100)); 
    next_frame_ = true;
  }
  stop_all_ = true;
  std::cout << "Exiting Viewer thread" << std::endl;
  pthread_exit(NULL);
}

// Thread to load Mat into the openCV viewer
void *start_image_viewer(void *threadid){
	long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting Image Viewer, Thread ID : " << tid << std::endl;
  
  // VIEWER PARAMETER
  cv::Mat image_display_;
  bool retrieve_image_ = false;
  
  // VIEWER LOOP
  while (!stop_all_){
		retrieve_image_ = dm.getFrame(image_display_);
		if(retrieve_image_){
			cv::imshow("Image Viewer", image_display_);
			if(cv::waitKey(30) >= 0) std::cout << "Key out" << std::endl;
		}
		retrieve_image_ = false;
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
  point_cloud_sub_ = nh_.subscribe("kinect2/sd/points", 1, cloud_callback);
  
  //~ // SUBSCRIBE TO 2D IMAGE TOPIC
  std::cout << "Subscribing to Kinect Image Topic" << std::endl;
	image_transport::ImageTransport it(nh_);
  image_transport::Subscriber image_sub_ = it.subscribe("/kinect2/sd/image_color_rect", 1, image_callback);
  //~ image_transport::Subscriber image_sub_ = it.subscribe("/kinect2/hd/image_color", 1, image_callback);
  // CONTINUOUS RETRIEVAL
  while (!stop_all_){
		cb_queue_.callAvailable();
		ros::spinOnce();
	}
	
	image_sub_.shutdown();
	point_cloud_sub_.shutdown();
  ros::shutdown();
  
  //~ std::cout << "Size of point cloud: " << cloud_a->points.size() << std::endl;
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
  
  // CAMERA CALIBRATION
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  loadCalibrationMatrix("kinect_sd");
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
  thread_error_ = pthread_create(&thread[i], NULL, start_cloud_viewer, (void *)i);
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
  
  //~ // PREPARE IMAGE VIEWER CALLBACK
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
