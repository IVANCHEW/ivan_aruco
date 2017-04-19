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
int hsv_target2_;
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
int icp_max_iter_;
float icp_corr_distance_;
float plane_cut_;
int sor_mean_;
float sor_thresh_;

// For Model Reconstruction
float downsampling_size_;

#include "data_manager.cpp"

DataManagement dm;

void generateTestModelCloud(pcl::PointCloud<PointType>::Ptr &cloud){
	cloud->width = 4;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);
	
	for(int i=0; i<cloud->points.size(); i++){
		cloud->points[i].x = 0;
		cloud->points[i].y = 0;
		cloud->points[i].z = 0;
	}
	
	cloud->points[1].x = 0.08;
	cloud->points[2].y = 0.17;
	cloud->points[3].x = 0.015;
	cloud->points[3].y = 0.07;

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

// Thread to retrieve ROS messages from topics subscribed
void *start_ros_callback(void *threadid){
  long tid;
  tid = (long)threadid;
  ROS_DEBUG_STREAM("Starting ROS Callback Listener, Thread ID : ");
  ros::NodeHandle nh_, nh_private_;
  ros::CallbackQueue cb_queue_;
	
  // SUBSCRIBE TO POINT CLOUD TOPIC
  ROS_DEBUG("Subscribing to Kinect Point Cloud Topic");
  ros::Subscriber point_cloud_sub_;
  point_cloud_sub_ = nh_.subscribe(cloud_topic_, 1, cloud_callback);
  
  //~ // SUBSCRIBE TO 2D IMAGE TOPIC
  ROS_DEBUG("Subscribing to Kinect Image Topic");
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
  
  ROS_DEBUG("Exiting ROS callback thread");
  pthread_exit(NULL);
}

// PCL VIEWER KEYBOARD CALL BACK
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void){
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  std::string key_value = event.getKeySym();
  std::cout << key_value << std::endl;
  
  if (event.getKeySym () == "d" && event.keyDown ()){
		std::cout << "Key out, d" << std::endl;
		ROS_DEBUG("PCL Viewer key down, input value = 1");
    input_value_ = 1;
  }
  else if (event.getKeySym () == "s" && event.keyDown ()){
		std::cout << "Key out, s" << std::endl;
		ROS_DEBUG("PCL Viewer key down, input value = 1");
    input_value_ = 2;
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
  viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
  viewer.setCameraPosition(0.0308721, 0.0322514, -1.05573, 0.0785146, -0.996516, -0.0281465);
  viewer.setPosition(49, 540);
  
  // RECONSTRUCTION VIEWER
  pcl::visualization::PCLVisualizer viewer2("Reconstruction Viewer");
  viewer2.setCameraPosition(0.0308721, 0.0322514, -1.05573, 0.0785146, -0.996516, -0.0281465);
  viewer2.setPosition(958, 52);
  
  pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr highlight_cloud (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr highlight_markers_cloud_ (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_cloud_ (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr transformed_cloud_ (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr reconstructed_cloud_ (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr reconstructed_markers_cloud_ (new pcl::PointCloud<PointType> ());
  pcl::visualization::PointCloudColorHandlerCustom<PointType> highlight_color_handler (highlight_cloud, 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> transformed_color_handler_ (transformed_cloud_, 0, 0, 255);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler_ (model_cloud_, 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> marker_color_handler_ (reconstructed_markers_cloud_, 255, 0, 0);
 
  bool first_cloud_ = true;
  bool first_stitch_ = true;
  bool first_marker_stitch_ = true;
  bool retrieve_cloud_ = false;
  bool retrieve_highlight_ = false;
  bool retrieve_markers_highlight_ = false;
  bool retrieve_index_ = false;
  std::vector<int> scene_corrs, database_corrs;
	generateTestModelCloud(model_cloud_);
  dm.loadDatabaseDescriptors(model_cloud_);
  
  // VIEWER LOOP
  while (!stop_all_){
		current_time_ = ros::Time::now().toSec();
		
		Eigen::Matrix4f estimated_pose_;
		
		bool pose_found_;
		
		// NODE TERMINATION CONDITION
		if(current_time_ - last_retrieved_time_ > 5.0){
			std::cout << "Terminating because of frame retrieval delay" << std::endl;
			stop_all_  = true;
		}
		
		// PROCESS IMAGE
		if(dm.getCloudAndImageLoadStatus()){
			next_frame_ = false;
			// DETECT POSE ESTIMATION MARKERS
			ROS_DEBUG("Detecting Markers");
			dm.detectMarkers(blur_param_, hsv_target_, hsv_threshold_ , contour_area_min_, contour_area_max_, contour_ratio_min_, contour_ratio_max_, aruco_detection_);
			
			ROS_DEBUG("Computing Descriptors");
			dm.computeDescriptors();
			ROS_DEBUG("Finding Matching Descriptors");
			dm.getMatchingDescriptor();
			ROS_DEBUG("Beginning Pose Estimation");
			pose_found_ = dm.computePoseEstimate(estimated_pose_, gc_size_, gc_threshold_);
			
			ROS_DEBUG("Retrieving Image and Cloud");
			retrieve_image_ = dm.getFrame(image_display_);
			retrieve_cloud_ = dm.getCloud(cloud);
			retrieve_highlight_ = dm.getHighlightCloud(highlight_cloud);
			// OBTAIN CORRESPONDENCE
			if(pose_found_){
				ROS_DEBUG("Pose Found, Retrieving Correspondence");
				dm.getCorrespondence(scene_corrs, database_corrs);
				pcl::transformPointCloud (*model_cloud_, *transformed_cloud_, estimated_pose_);
			}
			
			dm.clearPixelPoints();
			dm.clearDescriptors();
			
			// DETECT RECONSTRUCTED MARKERS
			ROS_DEBUG("Detecting Reconstructed Markers");
			dm.detectMarkers(blur_param_, hsv_target2_, hsv_threshold_ , contour_area_min_, contour_area_max_, contour_ratio_min_, contour_ratio_max_, aruco_detection_);
			ROS_DEBUG("Retrieving Marker Highlights");
			retrieve_markers_highlight_ = dm.getHighlightCloud(highlight_markers_cloud_);
			ROS_DEBUG("Clearing Variables");
			dm.clearPixelPoints();
			dm.clearDescriptors();
			dm.clearFrameAndCloud();
			next_frame_ = true;
			ROS_DEBUG("End of Image Processing");
		}
		
		// DISPLAY IMAGE
		if(retrieve_image_){
			ROS_DEBUG("Displaying RGB Image");
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
				ROS_INFO("Initialising New Clouds");
				
				// ADD SCENE CLOUD FORM KINECT
				ROS_DEBUG_STREAM("Adding First Scene, Cloud size: " << cloud->points.size());
				viewer.addPointCloud(cloud, "cloud");

				// ADD MODEL CLOUD
				ROS_DEBUG_STREAM("Adding First Model Cloud, Cloud size: " << model_cloud_->points.size());
				viewer.addPointCloud(model_cloud_, model_color_handler_, "model");
				viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "model");

				first_cloud_ = false;
			}
			else{
				ROS_DEBUG("Updating Viewer 1 Point Clouds");
				
				// UPDATING SCENE CLOUD FROM KINECT
				viewer.updatePointCloud(cloud, "cloud");
				
				if(pose_found_){
					// UPDATING TRANSFORMED CLOUD
					ROS_DEBUG("Updating Transformed Cloud");
					if(!viewer.updatePointCloud(transformed_cloud_, transformed_color_handler_, "transformed")){
						ROS_DEBUG("First Transform Cloud, adding...");
						viewer.addPointCloud(transformed_cloud_, transformed_color_handler_, "transformed");
						viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "transformed");
					}
					
					// UPDATING CORRESPONDENCE LINE
					ROS_DEBUG("Updating Correspondence Lines");
					viewer.removeAllShapes();
					for(int i=0; i<scene_corrs.size(); i++){
						std::stringstream ss;
						ss << i;
						viewer.addLine<PointType, PointType> (model_cloud_->points[database_corrs[i]], highlight_cloud->points[scene_corrs[i]], 0, 255, 0, ss.str());
					}
				}
			}
			
			// SPIN VIEWER 1 AFTER UPDATE
			ROS_DEBUG("Spinning Viewer 1");
			viewer.spinOnce();
			ROS_DEBUG("Viewer 1 Spun");
			
			// ADD VIEW TO RECONSTRUCTED CLOUD
			if(pose_found_ && input_value_==1){
				ROS_INFO("Adding view to reconstructed cloud");
				pcl::PointCloud<PointType>::Ptr add_cloud_ (new pcl::PointCloud<PointType> ()); 	
				pcl::PointCloud<PointType>::Ptr add_cloud_filtered_ (new pcl::PointCloud<PointType> ()); 	
				pcl::PointCloud<PointType>::Ptr add_markers_cloud_ (new pcl::PointCloud<PointType> ()); 	
				
				// STEP 1: TRANSFORM THE SCENE TO THE REFERENCE DATUM
				Eigen::Matrix4f pose_inverse_;
				pose_inverse_ = estimated_pose_.inverse();
				pcl::PointCloud<PointType>::Ptr datum_scene_cloud_ (new pcl::PointCloud<PointType> ()); 
				pcl::transformPointCloud<PointType> (*cloud, *datum_scene_cloud_, pose_inverse_);
				pcl::transformPointCloud<PointType> (*highlight_markers_cloud_, *add_markers_cloud_, pose_inverse_);
				
				// STEP 2: OBTAIN A REFERENCE PLANE
				pcl::SACSegmentation<pcl::PointXYZ> seg;
				pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
				pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
				seg.setOptimizeCoefficients (true);
				seg.setModelType (pcl::SACMODEL_PLANE);
				seg.setMethodType (pcl::SAC_RANSAC);
				seg.setDistanceThreshold (0.005);
				seg.setInputCloud (model_cloud_);
				seg.segment (*inliers, *coefficients);
				if (inliers->indices.size () == 0)
				{
					ROS_ERROR ("Could not estimate a planar model for the given dataset.");
				}
				else{
					// STEP 3: REMOVE THE PLANE AND ALL POINTS BELOW IT
					float z_plane;
					
					for (int i=0; i<=datum_scene_cloud_->points.size(); ++i)
					{
						 z_plane = (-coefficients->values[3] - coefficients->values[0]*cloud->points[i].x - coefficients->values[1]*cloud->points[i].y)/ coefficients->values[2];
						 if (datum_scene_cloud_->points[i].z < z_plane - plane_cut_)
							add_cloud_->points.push_back (datum_scene_cloud_->points[i]);
					}        
				}
				
				// STEP 4: PERFORM STATISTICAL OUTLIER FILTER
				pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outlier_filter_;
				outlier_filter_.setInputCloud (add_cloud_);
				outlier_filter_.setMeanK (sor_mean_);
				outlier_filter_.setStddevMulThresh (sor_thresh_);
				outlier_filter_.filter (*add_cloud_filtered_);
				
				// STEP 5: ADD CLOUD TO RECONSTRUCTED CLOUD
				*reconstructed_cloud_ += *add_cloud_filtered_;
				*reconstructed_markers_cloud_ += *add_markers_cloud_;
				
				// STEP 6: PERFORM DOWN SAMPLING
				pcl::VoxelGrid<PointType> sor;
				pcl::PointCloud<PointType>::Ptr downsampled_cloud_ (new pcl::PointCloud<PointType> ()); 
				sor.setInputCloud (reconstructed_cloud_);
				sor.setLeafSize (downsampling_size_,downsampling_size_,downsampling_size_);
				sor.filter (*downsampled_cloud_);
				reconstructed_cloud_ = downsampled_cloud_;
				
				input_value_=0;
				
				if(first_stitch_){
					viewer2.addPointCloud(reconstructed_cloud_, "reconstruction");
					first_stitch_=false;
				}else{
					viewer2.updatePointCloud(reconstructed_cloud_, "reconstruction");
				}
				
				if(first_marker_stitch_){
					viewer2.addPointCloud(reconstructed_markers_cloud_, marker_color_handler_, "marker reconstruction");
					viewer2.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "marker reconstruction");
					first_marker_stitch_=false;
				}else{
					viewer2.updatePointCloud(reconstructed_markers_cloud_, marker_color_handler_, "marker reconstruction");
				}
				
				viewer2.spinOnce();
			}
			
			// SAVE RECONSTRUCTED CLOUD
			if(input_value_==2){
				ROS_INFO("Saving reconstructed cloud");
				
				// STEP 2: WRITE CLOUD TO FILE
				pcl::PCDWriter writer;
				writer.write<PointType> (package_path_ + "/reconstruced.pcd", *reconstructed_cloud_, false);
				writer.write<PointType> (package_path_ + "/markers.pcd", *reconstructed_markers_cloud_, false);
				input_value_=0;
				stop_all_=true;
			}
			
		}

		retrieve_cloud_ = false;
		retrieve_image_ = false;
		retrieve_index_ = false;
		ROS_DEBUG("End of Viewer Loop");
	}
	std::cout << "Exiting Image Viewer Thread" << std::endl;
	pthread_exit(NULL);
}

int main (int argc, char** argv){
  ROS_DEBUG("Kinect Model Reconstruction Package");
  ros::init(argc, argv, "kinect_model_reconstruction");
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_("~");
  
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
    
  nh_private_.getParam("blur_param_", blur_param_);
  nh_private_.getParam("hsv_target_", hsv_target_);
  nh_private_.getParam("hsv_target2_", hsv_target2_);
  nh_private_.getParam("hsv_threshold_", hsv_threshold_);
  nh_private_.getParam("contour_area_min_", contour_area_min_);
  nh_private_.getParam("contour_area_max_", contour_area_max_);
  nh_private_.getParam("contour_ratio_min_", contour_ratio_min_);
  nh_private_.getParam("contour_ratio_max_", contour_ratio_max_);
  nh_private_.getParam("aruco_detection_", aruco_detection_);
  nh_private_.getParam("image_topic_", image_topic_);
  nh_private_.getParam("cloud_topic_", cloud_topic_);
  nh_private_.getParam("desc_match_thresh_", desc_match_thresh_);
  nh_private_.getParam("gc_size_", gc_size_);
  nh_private_.getParam("gc_threshold_", gc_threshold_);
  nh_private_.getParam("downsampling_size_", downsampling_size_);
  nh_private_.getParam("icp_max_iter_", icp_max_iter_);
  nh_private_.getParam("icp_corr_distance_", icp_corr_distance_);
  nh_private_.getParam("plane_cut_", plane_cut_);
  nh_private_.getParam("sor_mean_", sor_mean_);
  nh_private_.getParam("sor_thresh_", sor_thresh_);
  nh_private_.getParam("debug_", debug_);
  
  if (debug_)
	{
		ROS_INFO("Debug Mode ON");
		pcl::console::setVerbosityLevel(pcl::console::L_INFO);
		ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
	}
	else
	{
		ROS_INFO("Debug Mode OFF");
		pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
		ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	}
	dm.setDescMatchThreshold(desc_match_thresh_);
	
  // CAMERA CALIBRATION
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  loadCalibrationMatrix("kinect_sd");
	focal_length = camera_matrix.at<double>(0,0);
	dm.setParameters(2*camera_matrix.at<double>(1,2), 2*camera_matrix.at<double>(0,2), package_path_);
	dm.setCameraParameters(camera_matrix, dist_coeffs);
	dm.setIcpParameters(icp_max_iter_, icp_corr_distance_);
	
  // DEBUGGING
  ROS_DEBUG_STREAM("Package Path: " << package_path_);
  if(debug_){
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
	}
	
  // THREADING
  pthread_t thread[3];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  int thread_error_;
  int i=0;
  void *status;
  
  // PREPARE VIEWER THREAD
	thread_error_ = pthread_create(&thread[i], NULL, start_ros_callback, (void *)i);
  if (thread_error_){
    ROS_ERROR_STREAM("Error:unable to create ROS Callback thread," << thread_error_);
    exit(-1);
  }
  
  // PREPARE ROS CALLBACK
  i++;
	thread_error_ = pthread_create(&thread[i], NULL, start_viewer, (void *)i);
  if (thread_error_){
    ROS_ERROR_STREAM("Error:unable to create viewer thread," << thread_error_);
    exit(-1);
  }

  // INITIATE THREADS
  pthread_attr_destroy(&attr);
  
  //initially j < 1
  for(int j=0; j < 1; j++ ){
    thread_error_ = pthread_join(thread[i], &status);
    if (thread_error_){
      ROS_ERROR_STREAM("Error:unable to join," << thread_error_);
      exit(-1);
    }
  }
  
  ROS_INFO("End of main script");
  
  return 0;
}
