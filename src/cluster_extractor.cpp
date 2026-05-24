#include "get_seg_3d_coord/cluster_extractor.hpp"

#include <cstdint>
#include <utility>

#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <rclcpp/rclcpp.hpp>

namespace get_seg_3d_coord {

std::vector<ClusterResult> clusterPerBBox(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &accumulated_cloud,
    const std::vector<int> &per_point_bbox,
    const std::vector<BBoxCandidate> &candidates,
    const double cluster_tolerance,
    const int min_cluster_size,
    const int max_cluster_size,
    const rclcpp::Logger &logger) {
  std::vector<ClusterResult> results;
  if (candidates.empty() || accumulated_cloud->points.empty()) {
    return results;
  }

  std::vector<std::vector<int>> bins(candidates.size());
  for (std::size_t i = 0; i < per_point_bbox.size(); ++i) {
    const int bbox_index = per_point_bbox[i];
    if (bbox_index < 0 ||
        static_cast<std::size_t>(bbox_index) >= candidates.size()) {
      continue;
    }
    bins[static_cast<std::size_t>(bbox_index)].push_back(static_cast<int>(i));
  }

  for (std::size_t b = 0; b < candidates.size(); ++b) {
    const auto &indices = bins[b];
    if (static_cast<int>(indices.size()) < min_cluster_size) {
      RCLCPP_WARN(logger,
                  "local_search bbox[%zu] class=%s skipped: points=%zu < "
                  "min_cluster_size=%d",
                  b, candidates[b].class_id.c_str(), indices.size(),
                  min_cluster_size);
      continue;
    }

    auto sub_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    sub_cloud->points.reserve(indices.size());
    for (const int idx : indices) {
      sub_cloud->points.push_back(accumulated_cloud->points[idx]);
    }
    sub_cloud->width = static_cast<std::uint32_t>(sub_cloud->points.size());
    sub_cloud->height = 1;
    sub_cloud->is_dense = false;

    auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    tree->setInputCloud(sub_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> extractor;
    extractor.setClusterTolerance(cluster_tolerance);
    extractor.setMinClusterSize(min_cluster_size);
    extractor.setMaxClusterSize(max_cluster_size);
    extractor.setSearchMethod(tree);
    extractor.setInputCloud(sub_cloud);
    extractor.extract(cluster_indices);

    ClusterResult cr;
    cr.bbox_index = b;
    cr.class_id = candidates[b].class_id;
    cr.score = candidates[b].score;

    if (cluster_indices.empty()) {
      RCLCPP_WARN(logger,
                  "local_search bbox[%zu] class=%s PCL clustering yielded 0 "
                  "clusters from %zu points (tol=%.2f); using all points as "
                  "fallback",
                  b, candidates[b].class_id.c_str(), indices.size(),
                  cluster_tolerance);
      cr.indices.indices = indices;
    } else {
      std::size_t dominant = 0;
      for (std::size_t k = 1; k < cluster_indices.size(); ++k) {
        if (cluster_indices[k].indices.size() >
            cluster_indices[dominant].indices.size()) {
          dominant = k;
        }
      }
      cr.indices.indices.reserve(cluster_indices[dominant].indices.size());
      for (const int sub_idx : cluster_indices[dominant].indices) {
        cr.indices.indices.push_back(indices[static_cast<std::size_t>(sub_idx)]);
      }
      RCLCPP_INFO(logger,
                  "local_search bbox[%zu] class=%s clusters=%zu dominant=%zu",
                  b, candidates[b].class_id.c_str(), cluster_indices.size(),
                  cluster_indices[dominant].indices.size());
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*accumulated_cloud, cr.indices.indices, centroid);
    cr.center_lidar =
        Eigen::Vector3f(centroid.x(), centroid.y(), centroid.z());
    results.push_back(std::move(cr));
  }
  return results;
}

}  // namespace get_seg_3d_coord
