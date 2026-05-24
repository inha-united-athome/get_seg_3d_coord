#pragma once

#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/logger.hpp>

#include "get_seg_3d_coord/local_search_types.hpp"

namespace get_seg_3d_coord {

std::vector<ClusterResult> clusterPerBBox(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &accumulated_cloud,
    const std::vector<int> &per_point_bbox,
    const std::vector<BBoxCandidate> &candidates,
    double cluster_tolerance,
    int min_cluster_size,
    int max_cluster_size,
    const rclcpp::Logger &logger);

}  // namespace get_seg_3d_coord
