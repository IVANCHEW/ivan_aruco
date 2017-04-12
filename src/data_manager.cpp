// IMAGE PROCESSING
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>

// POINT CLOUD PROCESSING
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/console/print.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI IntensityType;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::PointNormal NormalPointType;

class DataManagement
{
	private:
	
		cv::Mat tvec;
		cv::Mat rvec;
		cv::Mat camera_matrix;
		cv::Mat dist_coeffs;
		cv::Mat frame;
		float aruco_size = 0.08/2;
		float desc_match_thresh_ = 0.005;
		cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
		std::vector <int> point_id;
		std::vector <int> cloud_index;
		std::vector <int> correspondence_point_;
		std::vector <int> correspondence_database_;
		std::vector < std::vector < float > > feature_desc, database_desc_, test_desc_;
		std::vector < std::vector < int > > feature_desc_index, database_desc_index_, test_desc_index_;
		std::vector <cv::Point2f> pixel_position_;
		std::string package_path_;
		int index_count = 0;
		int frame_height, frame_width;
		bool transformation_ready = false;
		bool image_ready = false;
		bool cloud_ready = false;
		bool pixel_point_ready = false;
		bool reading_pixel_point = false;
		bool parameters_ready = false;
		bool camera_parameters_ready = false;
		bool database_desc_ready_ = false;
		pcl::PointCloud<PointType>::Ptr cloud;
		
	public:
	
		void setParameters(double h, double w, std::string p);
		void setCameraParameters(cv::Mat c, cv::Mat d);
		void setPixelPointReady();
	
		void loadTransform(cv::Mat t, cv::Mat r);
		void loadFrame(cv::Mat f);
		void loadCloud(pcl::PointCloud<PointType>::Ptr &c);
		void loadPixelPoint(cv::Point2f p, int id);
		void loadDatabaseDescriptors(std::vector < std::vector < int > > index_vector, std::vector < std::vector < float > > element_vector);
		// To be deleted
		void loadTestDescriptors();
		
		void getTransform(cv::Mat& t, cv::Mat& r);
		bool getFrame(cv::Mat& f);
		bool getCloud(pcl::PointCloud<PointType>::Ptr &r);
		bool getPixelPoint(cv::Point2f &p);
		bool getPointCloudIndex(int &index, int n);
		bool getPointIndexSize(int &n);
		bool getCloudAndImageLoadStatus();
		bool getImageLoadStatus();
		bool getMatchingDescriptor();
		
		void computeDescriptors();
		
		void arrangeDescriptorsElements();
		
		void clearPixelPoints();
		void clearDescriptors();
		
		bool statusTransform();
		
		void labelMarkers();
		
		void printDescriptors();
		void printDatabaseDescriptors();
		
		bool detectMarkers(int blur_param_, int hsv_target_, int hsv_threshold_ , int contour_area_min_, int contour_area_max, double contour_ratio_min_, double contour_ratio_max_, bool aruco_detection_);
		
		// Circular Marker Detection Function
		bool circleEstimation (cv::Mat& input_image, int blur_param_, int hsv_target_, int hsv_threshold_ , int contour_area_min_, int contour_area_max_,  double contour_ratio_min_, double contour_ratio_max_){
			
			int marker_count_ = 0;
			
			//#1 Get Image Shape Parameters
			//std::cout << "Step 1: Getting Image Shape Parameters" << std::endl;
			int row = input_image.rows;
			int col = input_image.cols;
			//~ cv::imwrite(package_path_ + "/pose_estimation_frames/original_image.png", input_image);
			
			//#2 Median Blur Image
			cv::Mat image_blurred;
			cv::medianBlur(input_image, image_blurred, blur_param_);
			//~ cv::imwrite(package_path_ + "/pose_estimation_frames/blurred_image.png", image_blurred);
			
			//#3 Apply HSV Filtering
			cv::Mat image_hsv;
			cv::Mat image_hsv_filtered;
			cv::cvtColor(image_blurred, image_hsv, CV_BGR2HSV);
			cv::inRange(image_hsv,cv::Scalar(hsv_target_ - hsv_threshold_,0,0), cv::Scalar(hsv_target_ + hsv_threshold_,255,255),image_hsv_filtered);
			//~ cv::imwrite(package_path_ + "/pose_estimation_frames/hsv_image.png", image_hsv);
			//~ cv::imwrite(package_path_ + "/pose_estimation_frames/hsv_filtered_image.png", image_hsv_filtered);
			
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
						
						//#7 Compute Centroid and give temporary ID
						if(marker_confirmed_){
							//~ std::cout << "Id: " << marker_count_ << ", area: " << contour_area << std::endl;
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
							loadPixelPoint(p, marker_count_);
							marker_count_++;
						}
					}
			}
			//~ cv::imwrite(package_path_ + "/pose_estimation_frames/contour_marked_image.png", input_image);
			
			//#8 Return True if sufficient markers are present to make pose estimate
			if (marker_count_ >= 3){
				setPixelPointReady();
				return true;
			}else{
				return false;
			}
		}

		// ARUCO Marker Detection function
		bool arucoPoseEstimation(cv::Mat& input_image, int id, cv::Mat& tvec, cv::Mat& rvec, cv::Mat& mtx, cv::Mat& dist, bool draw_axis){
			// Contextual Parameters
			//std::cout << std::endl << "Pose estimation called..." << std::endl;
			float aruco_square_size = aruco_size*2;
			bool marker_found = false;
			std::vector< int > marker_ids;
			std::vector< std::vector<cv::Point2f> > marker_corners, rejected_candidates;
			cv::Mat gray;
			
			cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
			cv::aruco::detectMarkers(gray, dictionary, marker_corners, marker_ids);	
			clearPixelPoints();
			std::cout << "Number of markers detected: " << marker_ids.size() << std::endl;
			if (marker_ids.size() > 0){
				for (int i = 0 ; i < marker_ids.size() ; i++){
					std::cout << "Marker ID found: " << marker_ids[i] << std::endl;
					
					std::vector< std::vector<cv::Point2f> > single_corner(1);
					single_corner[0] = marker_corners[i];
					
					for (int j = 0; j < 4; j++){
						loadPixelPoint(marker_corners[i][j], marker_ids[i]);
					}
					
					cv::aruco::estimatePoseSingleMarkers(single_corner, aruco_square_size, mtx, dist, rvec, tvec);
					if (draw_axis && camera_parameters_ready){
						std::cout << "Drawing markers and axis" << std::endl;
						cv::aruco::drawDetectedMarkers(input_image, marker_corners, marker_ids);
						cv::aruco::drawAxis(input_image, mtx, dist, rvec, tvec, aruco_square_size/2);
					}
				}
				setPixelPointReady();
				marker_found = true;
			}
			else{
				std::cout << "No markers detected" << std::endl;
			}
			
			return marker_found;
}
};

void DataManagement::setParameters(double h, double w, std::string p){
	package_path_ = p;
	frame_height = (int) h;
	frame_width = (int) w;
	
	std::cout << std::endl << "Parameters set. W = " << frame_width << " H = " << frame_height << std::endl;
	
	parameters_ready = true;
}

void DataManagement::setCameraParameters(cv::Mat c, cv::Mat d){
	camera_matrix = c;
	dist_coeffs = d;
	camera_parameters_ready = true;
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
	if (!image_ready){
		frame = f.clone();
		image_ready = true;
	}
}

void DataManagement::loadCloud(pcl::PointCloud<PointType>::Ptr &c){
	if (!cloud_ready){
		cloud = c;
		cloud_ready = true;
	}
}

// Also computes corresponding cloud index with the given pixel position
void DataManagement::loadPixelPoint(cv::Point2f p, int id){
	//~ std::cout << "Loading pixel point " << index_count << " . x: " << p.x << ", y: " << p.y << std::endl;
	int index_temp = (p.y) * frame_width + p.x;
	cloud_index.push_back(index_temp);
	point_id.push_back(id);
	pixel_position_.push_back(p);
	index_count++;
}

void DataManagement::loadDatabaseDescriptors(std::vector < std::vector < int > > index_vector, std::vector < std::vector < float > > element_vector){
	database_desc_index_.swap(index_vector);
	database_desc_.swap(element_vector);
	database_desc_ready_ = true;
	std::cout << "Database Descriptors Loaded" << std::endl;
	printDatabaseDescriptors();
}

void DataManagement::loadTestDescriptors(){
	std::vector < int > index;
	std::vector < float > desc;
	
	index.push_back(1);
	index.push_back(2);
	desc.push_back(0.08);
	desc.push_back(0.2);
	test_desc_.push_back(desc);
	test_desc_index_.push_back(index);
	loadDatabaseDescriptors(test_desc_index_, test_desc_);
	
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
		//~ std::cout << "Point cloud index returned: " << index << std::endl;
		return true;
	}
	else{
		return false;
	}
}

bool DataManagement::getPointIndexSize(int &n){
		n = point_id.size();
		//~ std::cout << "Point index size returned: " << n << std::endl;
		if (n>0){
			return true;
		}
		else{
			return false;
		}
}

bool DataManagement::getCloudAndImageLoadStatus(){
	if(image_ready && cloud_ready){
		return true;
	}
	else{
		return false;
	}
}

bool DataManagement::getImageLoadStatus(){
	return image_ready;
}

bool DataManagement::getMatchingDescriptor(){
	if (pixel_point_ready && database_desc_ready_){
		std::vector < int > match_point_;
		std::vector < int > match_database_;
		bool match_found_ = false;
		for (int n=0; n<feature_desc_index.size(); n++){
			int match_count_ = 0;
			for (int m=0; m<database_desc_index_.size(); m++){
				std::cout << "Checking feature: " << n << " with Database: " << m << std::endl;
				std::vector < int > ().swap(match_point_);
				std::vector < int > ().swap(match_database_);
				match_point_.push_back(n);
				match_database_.push_back(m);
				int j = 0;
				int count = 0;
				for (int i=0; i< feature_desc_index[n].size() ; i++){
					if ((feature_desc[n][i]>=database_desc_[m][j] - desc_match_thresh_) && (feature_desc[n][i]<=database_desc_[m][j] + desc_match_thresh_)){
						std::cout << "Element Matched" << std::endl;
						match_point_.push_back(feature_desc_index[n][i]);
						match_database_.push_back(database_desc_index_[m][j]);
						j++;
						count++;
					}
					else if (feature_desc[n][i] >= database_desc_[m][j] + desc_match_thresh_){
						std::cout << "Next database element" << std::endl;
						j++;
					}
					else{
						std::cout << "Not a match" << std::endl;
						break;
					}
				}
				if (count == 2){
					std::cout << "Match Found" << std::endl;
					match_found_ = true;
					break;
				}
			}
			if (match_found_)
				break;
		}
		if (match_found_){
			std::cout << "Descriptor Match Found" << std::endl;
			correspondence_point_.swap(match_point_);
			correspondence_database_.swap(match_database_);
			//~ labelMarkers();
		}else{
			std::cout << "No Match Found" << std::endl;
		}
	}else if(!pixel_point_ready){
		//~ std::cout << "Pixel points not ready, cannot determine match" << std::endl;
	}
	else if(!database_desc_ready_){
		//~ std::cout << "Database Descriptors not loaded, cannot determine match" << std::endl;
	}
}

void DataManagement::computeDescriptors(){
	if(pixel_point_ready && cloud_ready){
		int n = point_id.size();
		//~ std::cout << "Number of features detected: " << n << std::endl;
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
		
		//~ printDescriptors();
		//~ std::cout << "Finished computing descriptors... " << std::endl;
	}
	else if(!pixel_point_ready){
		//~ std::cout << "Pixel points not ready, cannot compute descriptor" << std::endl;
	}
	else if(!cloud_ready){
		//~ std::cout << "Point cloud not ready, cannot compute descriptor" << std::endl;
	}
}

void DataManagement::arrangeDescriptorsElements(){
	 
	if(pixel_point_ready && cloud_ready){
		//~ std::cout << std::endl << "Sorting descriptors according to magnitude..." << std::endl;
		for (int n=0; n < feature_desc.size() ; n++){
			//~ std::cout << "n: " << n << std::endl;
			bool lowest_score = true;
			int front_of_index;
			std::vector <float> sorted_desc;
			std::vector <int> sorted_index;
			//~ std::cout << "Pushback first index" << std::endl;
			sorted_desc.push_back(feature_desc[n][0]);
			sorted_index.push_back(feature_desc_index[n][0]);
			for (int i=1; i<feature_desc[n].size(); i++){    
				//~ std::cout << "i: " << i << std::endl;
				lowest_score = true;  
				for (int  j=0 ; j < sorted_desc.size() ; j++){   
					//~ std::cout << "j: " << j << std::endl;
					if(feature_desc[n][i] <= sorted_desc[j]){
						//~ std::cout << "Smaller than index j" << std::endl;
						front_of_index = j;
						lowest_score = false;
						break;
					}       
				}
				if (lowest_score==false){
					//~ std::cout << "Not lowest score" << std::endl;
					sorted_desc.insert(sorted_desc.begin() + front_of_index, feature_desc[n][i]);     
					sorted_index.insert(sorted_index.begin() + front_of_index, feature_desc_index[n][i]);           
				}
				else if(lowest_score==true){
					//~ std::cout << "Lowest score" << std::endl;
					sorted_desc.push_back(feature_desc[n][i]);
					sorted_index.push_back(feature_desc_index[n][i]);
				}
			}
			
			feature_desc_index[n].swap(sorted_index);
			feature_desc[n].swap(sorted_desc);
			//~ std::cout << "Descriptors Sorted..." << std::endl;
		}
		printDescriptors();
	}else if(!pixel_point_ready){
		//~ std::cout << "Pixel points not ready, cannot compute descriptor" << std::endl;
	}
	else if(!cloud_ready){
		//~ std::cout << "Point cloud not ready, cannot compute descriptor" << std::endl;
	}
}

void DataManagement::clearPixelPoints(){
	std::vector <int> ().swap(point_id);
	std::vector <int> ().swap(cloud_index);
	std::vector <cv::Point2f> ().swap(pixel_position_);
	index_count = 0;
	pixel_point_ready = false;
}

void DataManagement::clearDescriptors(){
	std::vector < std::vector < float > > ().swap(feature_desc);
	std::vector < std::vector < int > > ().swap(feature_desc_index);
	//~ std::cout << "Descriptors cleared" << std::endl;
}

bool DataManagement::statusTransform(){
	return transformation_ready;
}

void DataManagement::labelMarkers(){
	for(int i=0; i<correspondence_point_.size(); i++){
		for(int j=0; j<point_id.size(); j++){
			if(point_id[j]==correspondence_point_[i]){
				std::stringstream convert;
				convert << correspondence_database_[i];
				std::string s;
				s = convert.str();
				cv::putText(frame, s, pixel_position_[j], CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8, false);
			}
		}
	}
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

void DataManagement::printDatabaseDescriptors(){
	for( int i = 0; i < database_desc_index_.size() ; i++){
		std::cout << " Descriptor (" << i << ") : ";
		for (int j = 0 ; j < database_desc_[i].size() ; j++){
			std::cout << database_desc_[i][j] << "(" << database_desc_index_[i][j] << "), ";
		}
		std::cout << std::endl;
	}
}

bool DataManagement::detectMarkers(int blur_param_, int hsv_target_, int hsv_threshold_ , int contour_area_min_, int contour_area_max , double contour_ratio_min_, double contour_ratio_max_, bool aruco_detection_){
	if(image_ready){
		bool marker_found = false;
		if(aruco_detection_){
			marker_found = arucoPoseEstimation(frame, 0, tvec, rvec, camera_matrix, dist_coeffs, true);
		}else{
			marker_found = circleEstimation(frame, blur_param_, hsv_target_, hsv_threshold_ , contour_area_min_, contour_area_max, contour_ratio_min_, contour_ratio_max_);
		}
		return marker_found;
	}else{
		std::cout << "Image not ready, cannot detect markers" << std::endl;
		return false;
	}
}
