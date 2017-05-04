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

// File system access
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

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

// For Image Processing
int blur_param_ = 5;
int hsv_target_ = 145;
int hsv_threshold_ = 10;
int contour_area_min_ = 500;
int contour_area_max_ = 4000;
double contour_ratio_min_ = 6;
double contour_ratio_max_ = 9;
float downsampling_size_; 
// For Software control
int input_value_ = 0;
bool debug_;
bool got_image_ = false;
bool got_cloud_ = false;
int image_buffer_ = 0;
bool aruco_detection_ = false;

// For Reconstruction
pcl::PointCloud<PointType>::Ptr reconstructed_cloud_ (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr reconstructed_markers_cloud_ (new pcl::PointCloud<PointType> ());
float clus_tolerance_;
int clus_size_min_, clus_size_max_;
bool view_image_;

#include "data_manager.cpp"
DataManagement dm;

void GetFilesInDirectory(std::vector<std::string> &out, const std::string &directory){
	DIR *dir;
	class dirent *ent;
	class stat st;

	dir = opendir(directory.c_str());
	while ((ent = readdir(dir)) != NULL) {
			const std::string file_name = ent->d_name;
			const std::string full_file_name = directory + "/" + file_name;

			if (file_name[0] == '.')
					continue;

			if (stat(full_file_name.c_str(), &st) == -1)
					continue;

			const bool is_directory = (st.st_mode & S_IFDIR) != 0;

			if (is_directory)
					continue;

			out.push_back(full_file_name);
	}
	closedir(dir);
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
  ros::init(argc, argv, "SLAM Reconstructor");
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_("~");
  std::string image_path_, cloud_path_;
	ROS_INFO("Debug Mode ON");
	pcl::console::setVerbosityLevel(pcl::console::L_INFO);
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
		
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
	nh_private_.getParam("downsampling_size_", downsampling_size_);
	nh_private_.getParam("clus_tolerance_", clus_tolerance_);
	nh_private_.getParam("clus_size_min_", clus_size_min_);
	nh_private_.getParam("clus_size_max_", clus_size_max_);
	nh_private_.getParam("view_image_", view_image_);
	
  // CAMERA CALIBRATION
	dm.setParameters(540, 960,package_path_);
	dm.setReducedResolution(4);
	dm.setMinMarkers(1);
	
  // DEBUGGING
  ROS_INFO_STREAM("Package Path: " << package_path_);
	ROS_DEBUG_STREAM("Reading image from: " << package_path_  + image_path_);
	ROS_DEBUG_STREAM("Reading cloud from: " << package_path_  + cloud_path_);
	
	// READ FILES FROM DIRECTORY
	std::vector<std::string> image_file_list_, cloud_file_list_;
	GetFilesInDirectory(image_file_list_, package_path_  + image_path_);
	GetFilesInDirectory(cloud_file_list_, package_path_  + cloud_path_);
	ROS_DEBUG_STREAM("Number of image files detected: " << image_file_list_.size());
	ROS_DEBUG_STREAM("Number of cloud files detected: " << cloud_file_list_.size());
	
	// SORT ARRAYS
	std::sort(image_file_list_.begin(), image_file_list_.end());
	std::sort(cloud_file_list_.begin(), cloud_file_list_.end());
	
	// LOOP THROUGH AVAILABLE DATA
	for(int m=0 ; m<cloud_file_list_.size(); m++){
	
		ROS_DEBUG_STREAM("Analyzing set: " << m);
		std::stringstream ss;
		ss << m;
		std::string str = ss.str();
		ROS_DEBUG_STREAM("Image: " << image_file_list_[m]);
		ROS_DEBUG_STREAM("Cloud: " << cloud_file_list_[m]);
		
		//ADD POINT CLOUD
		pcl::PCDReader reader;
		pcl::PointCloud<PointType>::Ptr cloud_load(new pcl::PointCloud<PointType>);
		//~ reader.read (package_path_ + cloud_path_ + "/cloud" + str + ".pcd", *cloud_load);
		reader.read (cloud_file_list_[m], *cloud_load);	
		dm.loadCloud(cloud_load);		
		//ADD IMAGE
		cv::Mat image;
		//~ image = cv::imread(package_path_ + image_path_ + "/" + str + ".png", CV_LOAD_IMAGE_COLOR); 
		image = cv::imread(image_file_list_[m], CV_LOAD_IMAGE_COLOR); 
		
		//REMOVE NAN PIXELS FROM RGB IMAGE
		ROS_DEBUG_STREAM("Number of points in point cloud: " << cloud_load->points.size());
		int valid_count_=0;
		int scale_ = 4;
		for (int i=0; i<cloud_load->points.size(); i++){
			//WHEN POINT IS NAN
			if (cloud_load->points[i].x>0){
				valid_count_++;
			}else{
				int pixel_x = i % (image.cols/scale_);
				int pixel_y = i / (image.cols/scale_);
				for(int j=0; j<scale_; j++){
					for(int k=0; k<scale_; k++){
						int scaled_x_ = pixel_x*scale_ + j;
						int scaled_y_ = pixel_y*scale_ + k;
						cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(scaled_x_,scaled_y_));
						color[0] = 0;
						color[1] = 0;
						color[2] = 0;
						image.at<cv::Vec3b>(cv::Point(scaled_x_,scaled_y_)) = color;
					}
				}
			}
		}
		ROS_DEBUG_STREAM("Number of valid points: " << valid_count_);	
		dm.loadFrame(image);
		
		//DETECT MARKERS
		ROS_DEBUG("Detecting Markers");
		dm.detectMarkers(blur_param_, hsv_target_, hsv_threshold_ , contour_area_min_, contour_area_max_, contour_ratio_min_, contour_ratio_max_, false);
		
		//OBTAIN RESULTS
		ROS_DEBUG("Obtaining Results");
		pcl::PointCloud<PointType>::Ptr cloud_a, highlight_cloud;
		cv::Mat image_display_;
		bool retrieve_cloud_ = false;
		bool retrieve_highlight_ = false;
		bool retrieve_image_ = false;
		
		retrieve_cloud_ = dm.getCloud(cloud_a);
		retrieve_highlight_ = dm.getHighlightCloud(highlight_cloud);
		retrieve_image_ = dm.getFrame(image_display_);
		
		//VIEW IMAGE
		if (dm.getFrame(image_display_) && view_image_){
			cv::imshow("Image Viewer", image_display_);
			if(cv::waitKey(0) >= 0) {
				ROS_DEBUG("Image View Keyout");
			}
		}
		
		//COMPILING CLOUDS
		ROS_DEBUG("Compiling Clouds");
		if (retrieve_cloud_){
			ROS_DEBUG("Adding scene cloud");
			*reconstructed_cloud_ += *cloud_a;
		}
		if (retrieve_highlight_){
			ROS_DEBUG_STREAM("Adding highlight cloud. Highlight points size: " << highlight_cloud->points.size());
			*reconstructed_markers_cloud_ += *highlight_cloud;
		}
		
		//PERFORM DOWN SAMPLING
		ROS_DEBUG("Downsampling Cloud");
		pcl::VoxelGrid<PointType> sor;
		pcl::PointCloud<PointType>::Ptr downsampled_cloud_ (new pcl::PointCloud<PointType> ()); 
		sor.setInputCloud (reconstructed_cloud_);
		sor.setLeafSize (downsampling_size_,downsampling_size_,downsampling_size_);
		sor.filter (*downsampled_cloud_);
		reconstructed_cloud_ = downsampled_cloud_;
				
		//CLEARING DATA MANAGER
		dm.clearPixelPoints();
		dm.clearFrameAndCloud();
	}
	
	// CLUSTER FILTERING
	pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
	tree->setInputCloud (reconstructed_markers_cloud_);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (clus_tolerance_); 
	ec.setMinClusterSize (clus_size_min_);
	ec.setMaxClusterSize (clus_size_max_);
	ec.setSearchMethod (tree);
	ec.setInputCloud (reconstructed_markers_cloud_);
	ec.extract (cluster_indices);
	
	pcl::PointCloud<PointType>::Ptr cloud_cluster (new pcl::PointCloud<PointType>);
	std::vector< std::vector<float > > center_coordinates_;
	int centroid_number_=0;
	for (int i = 0; i < cluster_indices.size() ; i++){
		ROS_DEBUG_STREAM("Cluster " << i << " size: " << cluster_indices[i].indices.size());
		//Filtering Criteria
		if (cluster_indices[i].indices.size() > clus_size_min_ && cluster_indices[i].indices.size() < clus_size_max_){
			// FIND CENTER OF EACH CLUSTER AND ADD CLUSTER TO POINT CLOUD
			int n = cluster_indices[i].indices.size();
			ROS_DEBUG_STREAM("Number of points: " << n);
			float sum_x_=0;
			float sum_y_=0;
			float sum_z_=0;
			for (int j = 0 ; j < cluster_indices[i].indices.size() ; j++) {
				ROS_DEBUG_STREAM("Summing individual point moments, x: " << reconstructed_markers_cloud_->points[cluster_indices[i].indices[j]].x);
				sum_x_ = sum_x_ + reconstructed_markers_cloud_->points[cluster_indices[i].indices[j]].x;
				sum_y_ = sum_y_ + reconstructed_markers_cloud_->points[cluster_indices[i].indices[j]].y;
				sum_z_ = sum_z_ + reconstructed_markers_cloud_->points[cluster_indices[i].indices[j]].z;
				ROS_DEBUG("Pushing point into cluster cloud");
				cloud_cluster->points.push_back (reconstructed_markers_cloud_->points[cluster_indices[i].indices[j]]); 
			}
			ROS_DEBUG("Center coordinate vector pushback");
			center_coordinates_.push_back(std::vector<float>());
			ROS_DEBUG_STREAM("Storing values into center coordinate vector, x: " << sum_x_/n);
			center_coordinates_[centroid_number_].push_back(sum_x_/n);
			center_coordinates_[centroid_number_].push_back(sum_y_/n);
			center_coordinates_[centroid_number_].push_back(sum_z_/n);
			centroid_number_++;
			//~ sum_x_ = sum_x_/n;
			//~ sum_y_ = sum_y_/n;
			//~ sum_z_ = sum_z_/n;
		}
	}
	
	ROS_DEBUG_STREAM("Creating Point Cloud with centroid values, size: " << center_coordinates_.size());
	pcl::PointCloud<PointType>::Ptr cloud_cluster_center_ (new pcl::PointCloud<PointType>);
	cloud_cluster_center_->width = center_coordinates_.size();
	cloud_cluster_center_->height = 1;
	cloud_cluster_center_->is_dense = false;
	cloud_cluster_center_->points.resize(cloud_cluster_center_->width * cloud_cluster_center_->height);
	ROS_DEBUG("Updating Point Cloud with centroid values");
	for(int i=0; i<centroid_number_; i++){
		cloud_cluster_center_->points[i].x = center_coordinates_[i][0];
		cloud_cluster_center_->points[i].y = center_coordinates_[i][1];
		cloud_cluster_center_->points[i].z = center_coordinates_[i][2];
	}
	
	
	ROS_DEBUG_STREAM("PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points.");
	cloud_cluster->width = cloud_cluster->points.size ();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;
	reconstructed_markers_cloud_ = cloud_cluster;
	
	// VIEWER PARAMETERS
	ROS_INFO("Starting Point Cloud Viewer...");
	viewer.addPointCloud(reconstructed_cloud_, "Reconstructed Model");
	pcl::visualization::PointCloudColorHandlerCustom<PointType> highlight_color_handler (reconstructed_markers_cloud_, 255, 0, 0);
	viewer.addPointCloud (reconstructed_markers_cloud_, highlight_color_handler, "Highlight Cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Highlight Cloud");
	pcl::visualization::PointCloudColorHandlerCustom<PointType> centroid_color_handler (cloud_cluster_center_, 0, 255, 0);
	viewer.addPointCloud (cloud_cluster_center_, centroid_color_handler, "Centroid Cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "Centroid Cloud");
	
	while (!viewer.wasStopped ()){
		viewer.spinOnce();
	}
	
	ros::shutdown();
	
  return 0;
}

