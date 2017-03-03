// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
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

// For Software control
int count;
int input_value_ = 0;
bool debug_;
//~ bool updateCloud = false;
//~ bool retrieve_cloud_ = false;
bool stop_all_ = false;

class DataManagement
{
	private:
	
		cv::Mat tvec;
		cv::Mat rvec;
		cv::Mat frame;
		bool transformation_ready = false;
		bool image_ready = false;
		bool cloud_ready = false;
		pcl::PointCloud<PointType>::Ptr cloud;
		
	public:
	
		void loadTransform(cv::Mat t, cv::Mat r);
		void loadFrame(cv::Mat f);
		void loadCloud(pcl::PointCloud<PointType>::Ptr &c);
		
		void getTransform(cv::Mat& t, cv::Mat& r);
		bool getFrame(cv::Mat& f);
		bool getCloud(pcl::PointCloud<PointType>::Ptr &r);
		
		bool statusTransform();
};

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

bool DataManagement::statusTransform(){
	return transformation_ready;
}

DataManagement dm;

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
  std::cout << "Cloud Callback Received" << std::endl;
  pcl::PCLPointCloud2 pcl_pc;
  pcl_conversions::toPCL(cloud_msg,pcl_pc);
  pcl::PointCloud<PointType>::Ptr scene_cloud(new pcl::PointCloud<PointType>);
  pcl::fromPCLPointCloud2(pcl_pc,*scene_cloud);
  dm.loadCloud(scene_cloud);
  //~ pcl::fromPCLPointCloud2(pcl_pc,*cloud_a);
  
  // Update viewer
  //~ if (updateCloud){
    //~ viewer.updatePointCloud(scene_cloud, "cloud_1");
    //~ viewer.updatePointCloud(cloud_a, "cloud a");
    //~ loadCloud
    //~ std::cout << "Frame: " << count << std::endl;
    //~ count++;
  //~ }
  //~ else{
    //~ viewer.addPointCloud (cloud_a,"cloud a");
    //~ updateCloud = true;
  //~ }
  //~ retrieve_cloud_ = true;
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){
	
	try{
		cv::Mat image;
		cv::Mat unDistort;
		cv::Mat rvec;
		cv::Mat tvec;
		bool marker_found;
		image = cv_bridge::toCvShare(msg, "bgr8")->image;
		cv::undistort(image, unDistort, camera_matrix, dist_coeffs);
		std::cout << "Image callback" << std::endl;
		std::cout << "Camera image size: " << unDistort.size() << std::endl;
	}
	catch(cv_bridge::Exception& e){
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
	
}

// THREAD FUNCTIONS
void *start_viewer(void *threadid){
  long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting Point Cloud Viewer, Thread ID : " << tid << std::endl;
  pcl::PointCloud<PointType>::Ptr cloud_a;
  bool update_ = false;
  bool retrieve_cloud_ = false;
  while (!viewer.wasStopped ()){
    retrieve_cloud_ = dm.getCloud(cloud_a);
    if (retrieve_cloud_){
			if (!update_){
				std::cout << "Added new cloud to viewer" << std::endl;
				viewer.addPointCloud(cloud_a, "cloud_a");
				update_ = true;
			}
			else{
				std::cout << "Updated cloud on viewer" << std::endl;
				viewer.updatePointCloud(cloud_a, "cloud_a");
			}
		}
    if (input_value_ == 1){
      viewer.close();
    }
    input_value_ == 0;
    viewer.spinOnce();
    boost::this_thread::sleep (boost::posix_time::microseconds (100)); 
  }
  stop_all_ = true;
  std::cout << "Exiting Viewer thread" << std::endl;
  pthread_exit(NULL);
}

void *start_ros_callback(void *threadid){
  long tid;
  tid = (long)threadid;
  std::cout << std::endl << "Starting ROS Callback Listener, Thread ID : " << tid << std::endl;
  ros::NodeHandle nh_, nh_private_;
  
  // SUBSCRIBE TO POINT CLOUD TOPIC
  std::cout << "Subscribing to Kinect Point Cloud Topic" << std::endl;
  ros::Subscriber point_cloud_sub_;
  point_cloud_sub_ = nh_.subscribe("kinect2/sd/points", 1, cloud_callback);
  
  // SUBSCRIBE TO 2D IMAGE TOPIC
  //~ std::cout << "Subscribing to Kinect Image Topic" << std::endl;
	//~ image_transport::ImageTransport it(nh_);
  //~ image_transport::Subscriber image_sub_ = it.subscribe("/kinect2/hd/image_color", 1, image_callback);
	//~ image_sub_.shutdown();
	
  // SINGLE CLOUD RETRIEVAL
  //~ double begin =ros::Time::now().toSec();
  //~ double current = ros::Time::now().toSec();
  //~ retrieve_cloud_ = false;
  //~ while (current-begin<5 && !retrieve_cloud_) {
    //~ current =ros::Time::now().toSec();
    //~ ros::spinOnce();
  //~ }
  
  // CONTINUOUS RETRIEVAL
  while (!stop_all_){
		ros::spinOnce();
	}
	
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
  count = 0;
  
  // CAMERA CALIBRATION
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  loadCalibrationMatrix("kinect");
	focal_length = camera_matrix.at<double>(0,0);
	
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
  pthread_t thread[2];
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
