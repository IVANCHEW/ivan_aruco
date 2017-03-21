// IMAGE PROCESSING
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

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
