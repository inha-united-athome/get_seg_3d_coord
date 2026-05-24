#pragma once

#include <cstddef>
#include <string>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <pcl/PointIndices.h>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace get_seg_3d_coord {

struct BBoxCandidate {
  std::string class_id;
  float score{0.0F};
  float u_min{0.0F};
  float u_max{0.0F};
  float v_min{0.0F};
  float v_max{0.0F};
};

struct ClusterResult {
  std::size_t bbox_index{0};
  std::string class_id;
  float score{0.0F};
  Eigen::Vector3f center_lidar{Eigen::Vector3f::Zero()};
  Eigen::Vector3f center_map{Eigen::Vector3f::Zero()};
  pcl::PointIndices indices;
};

struct SyncedInput {
  cv::Mat mask;
  std_msgs::msg::Header mask_header;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr bbox_msg;
};

}  // namespace get_seg_3d_coord
