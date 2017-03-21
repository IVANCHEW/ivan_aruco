cmake_minimum_required(VERSION 2.8.3)
project(ivan_aruco)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  eigen_conversions
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  tf_conversions
  image_transport
)

find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system thread)
find_package(PCL 1.7.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

catkin_package(CATKIN_DEPENDS
  pcl_ros
  tf_conversions
  image_transport
)

include_directories(${PCL_INCLUDE_DIRS}) 
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${OPENGL_INCLUDE_DIR}  ${GLUT_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(kinect_image_view src/kinect_image_view.cpp)
target_link_libraries(kinect_image_view yaml-cpp ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
add_dependencies(kinect_image_view ${catkin_EXPORTED_TARGETS})
 
add_executable(single_frame_overlay_viewer src/single_frame_overlay.cpp)
target_link_libraries(single_frame_overlay_viewer yaml-cpp ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
add_dependencies(single_frame_overlay_viewer ${catkin_EXPORTED_TARGETS})
 
add_executable(kinect_depth_test src/kinect_depth_test.cpp)
target_link_libraries(kinect_depth_test yaml-cpp ${PCL_LIBRARIES} ${catkin_LIBRARIES})
add_dependencies(kinect_depth_test ${catkin_EXPORTED_TARGETS})

add_executable(kinect_pose_estimation src/kinect_pose_estimation.cpp)
target_link_libraries(kinect_pose_estimation yaml-cpp ${PCL_LIBRARIES} ${catkin_LIBRARIES})
add_dependencies(kinect_pose_estimation ${catkin_EXPORTED_TARGETS})

add_executable(kinect_retrieve src/kinect_retrieve.cpp)
target_link_libraries(kinect_retrieve yaml-cpp ${PCL_LIBRARIES} ${catkin_LIBRARIES})
add_dependencies(kinect_retrieve ${catkin_EXPORTED_TARGETS})
