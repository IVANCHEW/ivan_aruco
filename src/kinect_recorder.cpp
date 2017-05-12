// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/service.h>
#include <ros/callback_queue.h>
#include <yaml-cpp/yaml.h>
#include <rosbag/bag.h>

// FOR IMAGE RECORDING
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "data_manager.cpp"

// IMAGE PROCESSING
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>

// POINT CLOUD PROCESSING
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>

DataManagement dm;
std::string package_path_;
std::string save_path_;
std::string bag_name_;
std::string point_cloud_topic_name_;
std::string image_topic_name_;
pcl::visualization::PCLVisualizer viewer("Kinect Viewer");

bool active_threads_ = true;
bool next_frame_ = true;

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

void image_callback(const sensor_msgs::ImageConstPtr& msg){
	std::cout << "Image callback" << std::endl;
	
	if(next_frame_){
		try{
			cv::Mat image;
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
  point_cloud_sub_ = nh_.subscribe(point_cloud_topic_name_, 1, cloud_callback);
  
  // SUBSCRIBE TO 2D IMAGE TOPIC
  std::cout << "Subscribing to Kinect Image Topic" << std::endl;
	image_transport::ImageTransport it(nh_);
  image_transport::Subscriber image_sub_ = it.subscribe(image_topic_name_, 1, image_callback);
  
  // CONTINUOUS RETRIEVAL
  while (active_threads_){
		cb_queue_.callAvailable();
		ros::spinOnce();
	}
	
	image_sub_.shutdown();
	point_cloud_sub_.shutdown();
  ros::shutdown();
  
  std::cout << "Exiting ROS callback thread" << std::endl;
  pthread_exit(NULL);
}

void *recorder(void *threadid){
	cv::namedWindow("Image Viewer",1);
	cv_bridge::CvImage out_msg;
	bool first_run_ = true;
	
	rosbag::Bag bag;
	bag.open(save_path_ + bag_name_, rosbag::bagmode::Write);
	
	while (active_threads_){
		
		if(dm.getCloudAndImageLoadStatus()){
			std::cout << "Begin Writing to bag" << std::endl;
			
			next_frame_ = false;
			
			// STEP 1 : RETRIEVE AND RECORD RGB IMAGE
			std::cout << "Begin writing image" << std::endl;
			cv::Mat image;
			dm.getRawFrame(image);
			
			// VISUALISATION
			cv::imshow("Image Viewer", image);
			if(cv::waitKey(30) >= 0){
				//~ std::cout << "Key out" << std::endl;
				active_threads_ = false;
			}
			
			// CONVERSION TO ROS MSG FOR SAVE
			out_msg.encoding = sensor_msgs::image_encodings::BGR8;
			out_msg.image    = image;
			
			// WRITING TO ROSBAG
			bag.write("image", ros::Time::now(), out_msg.toImageMsg());
			
			// STEP 2 : RETRIEVE AND RECORD POINT CLOUD
			std::cout << "Begin writing point cloud" << std::endl;
			pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
			dm.getCloud(cloud);
			
			// VISUALISATION
			if (first_run_){
				std::cout << "Added new cloud to viewer" << std::endl;
				viewer.addPointCloud(cloud, "cloud_a");
				first_run_ = false;
			}else{
				viewer.updatePointCloud(cloud, "cloud_a");
			}
			viewer.spinOnce();
    
			// CONVERSION TO ROS MSG FOR SAVE
			sensor_msgs::PointCloud2 cloud_msg_ros;
			pcl::toROSMsg(*cloud, cloud_msg_ros);
			
			// WRITING TO ROSBAG
			bag.write("cloud", ros::Time::now(), cloud_msg_ros);
			
			// SLIGHT DELAY
			boost::this_thread::sleep (boost::posix_time::microseconds (1000)); 
			next_frame_ = true;
			dm.clearFrameAndCloud();
		}
	}
	active_threads_ = false;
	bag.close();
	
	std::cout << "Exiting Image Viewer" << std::endl;
	pthread_exit(NULL);
}

int main (int argc, char** argv){
	std::cout << std::endl << "Kinect Record Started" << std::endl;
  ros::init(argc, argv, "kinect_record");
    
  // VARIABLE INITIALISATION
  package_path_ = ros::package::getPath("ivan_aruco");   
  //save_path_ = package_path_ + "/kinect_record_saves/";
  save_path_ = "/" + package_path_ + "/";
	// LOAD SINGLE IMAGE AND POINT CLOUD
	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_("~");
	nh_private_.getParam("bag_name_", 					bag_name_);
	nh_private_.getParam("point_cloud_topic_", 	point_cloud_topic_name_);
	nh_private_.getParam("image_topic_", 				image_topic_name_);
	
	// THREAD INITIALISATION	
	pthread_t thread[2];
	int threadError;
	int i=0;
	
	// START CV THREAD
	threadError = pthread_create(&thread[i], NULL, start_ros_callback, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}
	i++;
	
	threadError = pthread_create(&thread[i], NULL, recorder, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}

	pthread_exit(NULL);
	
	ros::shutdown();	
	
	std::cout << "Exiting Main Thread" << std::cout;
	return 0;
}
