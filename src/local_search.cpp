#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "get_seg_3d_coord/cluster_extractor.hpp"
#include "get_seg_3d_coord/local_search_types.hpp"
#include "get_seg_3d_coord/point_selector.hpp"
#include "get_seg_3d_coord/state_odometry.hpp"
#include "inha_interfaces/action/local_search.hpp"

class LocalSearchActionServer : public rclcpp::Node {
public:
  using LocalSearch = inha_interfaces::action::LocalSearch;
  using GoalHandle = rclcpp_action::ServerGoalHandle<LocalSearch>;

  LocalSearchActionServer()
      : Node("local_search"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    mask_topic_ = this->declare_parameter<std::string>(
        "mask_topic", "/sam2_segmentation_node/binary_mask/compressed");
    pointcloud_topic_ = this->declare_parameter<std::string>(
        "pointcloud_topic", "/livox/lidar");
    camera_info_topic_ = this->declare_parameter<std::string>(
        "camera_info_topic", "/camera/camera_head/color/camera_info");
    bbox_topic_ = this->declare_parameter<std::string>(
        "bbox_topic", "/detection/bbox_stream");
    odom_topic_ =
        this->declare_parameter<std::string>("odom_topic", "/odom");
    filtered_cloud_topic_ = this->declare_parameter<std::string>(
        "filtered_cloud_topic", "~/points");
    centers_topic_ = this->declare_parameter<std::string>(
        "centers_topic", "~/cluster_centers");
    clustered_cloud_topic_ = this->declare_parameter<std::string>(
        "clustered_cloud_topic", "~/clustered_points");
    markers_topic_ = this->declare_parameter<std::string>(
        "markers_topic", "~/cluster_markers");
    action_name_ =
        this->declare_parameter<std::string>("action_name", "local_search");
    camera_frame_param_ =
        this->declare_parameter<std::string>("camera_frame", "camera_head_color_optical_frame");
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    mask_threshold_ = this->declare_parameter<int>("mask_threshold", 0);
    cluster_tolerance_ =
        this->declare_parameter<double>("cluster_tolerance", 0.3);
    min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 3);
    max_cluster_size_ =
        this->declare_parameter<int>("max_cluster_size", 30000);
    input_sync_queue_size_ =
        this->declare_parameter<int>("input_sync_queue_size", 10);
    input_sync_slop_sec_ =
        this->declare_parameter<double>("input_sync_slop_sec", 0.2);
    tf_timeout_sec_ = this->declare_parameter<double>("tf_timeout_sec", 0.5);
    require_stable_odom_ =
        this->declare_parameter<bool>("require_stable_odom", false);
    odom_stable_window_sec_ =
        this->declare_parameter<double>("odom_stable_window_sec", 0.3);
    odom_stable_timeout_sec_ =
        this->declare_parameter<double>("odom_stable_timeout_sec", 1.0);
    max_relative_angular_velocity_rad_s_ =
        this->declare_parameter<double>("max_relative_angular_velocity_rad_s",
                                        0.20);
    odom_buffer_size_ =
        this->declare_parameter<int>("odom_buffer_size", 200);
    prefilter_cloud_ =
        this->declare_parameter<bool>("prefilter_cloud", false);
    prefilter_frame_ =
        this->declare_parameter<std::string>("prefilter_frame", "base");
    prefilter_x_min_ =
        this->declare_parameter<double>("prefilter_x_min", 0.0);
    prefilter_x_max_ =
        this->declare_parameter<double>("prefilter_x_max", 5.0);
    prefilter_y_min_ =
        this->declare_parameter<double>("prefilter_y_min", -2.0);
    prefilter_y_max_ =
        this->declare_parameter<double>("prefilter_y_max", 2.0);
    prefilter_z_min_ =
        this->declare_parameter<double>("prefilter_z_min", -0.05);
    prefilter_z_max_ =
        this->declare_parameter<double>("prefilter_z_max", 2.0);
    accumulate_frames_ =
        this->declare_parameter<int>("accumulate_frames", 10);
    accumulate_timeout_sec_ =
        this->declare_parameter<double>("accumulate_timeout_sec", 3.0);
    max_distance_from_reference_m_ = this->declare_parameter<double>(
        "max_distance_from_reference_m", 1.5);
    max_distance_from_reference_m_ =
        std::max(0.0, max_distance_from_reference_m_);
    accumulate_frames_ = std::max(1, accumulate_frames_);
    accumulate_timeout_sec_ = std::max(0.1, accumulate_timeout_sec_);
    input_sync_queue_size_ = std::max(1, input_sync_queue_size_);
    input_sync_slop_sec_ = std::max(0.0, input_sync_slop_sec_);
    odom_stable_window_sec_ = std::max(0.01, odom_stable_window_sec_);
    odom_stable_timeout_sec_ = std::max(0.01, odom_stable_timeout_sec_);
    max_relative_angular_velocity_rad_s_ =
        std::max(0.0, max_relative_angular_velocity_rad_s_);
    odom_buffer_size_ = std::max(2, odom_buffer_size_);
    state_odometry_.setMaxBufferSize(
        static_cast<std::size_t>(odom_buffer_size_));
    if (prefilter_x_min_ > prefilter_x_max_) {
      std::swap(prefilter_x_min_, prefilter_x_max_);
    }
    if (prefilter_y_min_ > prefilter_y_max_) {
      std::swap(prefilter_y_min_, prefilter_y_max_);
    }
    if (prefilter_z_min_ > prefilter_z_max_) {
      std::swap(prefilter_z_min_, prefilter_z_max_);
    }

    auto best_effort_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    mask_sub_.subscribe(this, mask_topic_, best_effort_qos.get_rmw_qos_profile());
    bbox_sub_.subscribe(this, bbox_topic_, reliable_qos.get_rmw_qos_profile());
    input_sync_ =
        std::make_shared<InputSynchronizer>(InputSyncPolicy(input_sync_queue_size_),
                                            mask_sub_, bbox_sub_);
    input_sync_->setMaxIntervalDuration(
        rclcpp::Duration::from_seconds(input_sync_slop_sec_));
    input_sync_->registerCallback(
        std::bind(&LocalSearchActionServer::syncedInputCallback, this,
                  std::placeholders::_1, std::placeholders::_2));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, reliable_qos,
        std::bind(&LocalSearchActionServer::cameraInfoCallback, this,
                  std::placeholders::_1));
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, best_effort_qos,
        std::bind(&LocalSearchActionServer::cloudCallback, this,
                  std::placeholders::_1));
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, reliable_qos,
        std::bind(&LocalSearchActionServer::odomCallback, this,
                  std::placeholders::_1));

    filtered_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            filtered_cloud_topic_, best_effort_qos);
    centers_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        centers_topic_, reliable_qos);
    clustered_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            clustered_cloud_topic_, reliable_qos);
    markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        markers_topic_, reliable_qos);

    action_server_ = rclcpp_action::create_server<LocalSearch>(
        this, action_name_,
        std::bind(&LocalSearchActionServer::handleGoal, this,
                  std::placeholders::_1, std::placeholders::_2),
        std::bind(&LocalSearchActionServer::handleCancel, this,
                  std::placeholders::_1),
        std::bind(&LocalSearchActionServer::handleAccepted, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "LocalSearch action node started. action=%s, mask=%s, "
                "cloud=%s, camera_info=%s, bbox=%s, odom=%s, "
                "accumulate_frames=%d, require_stable_odom=%s",
                action_name_.c_str(), mask_topic_.c_str(),
                pointcloud_topic_.c_str(), camera_info_topic_.c_str(),
                bbox_topic_.c_str(), odom_topic_.c_str(), accumulate_frames_,
                require_stable_odom_ ? "true" : "false");
  }

private:
  using MaskMsg = sensor_msgs::msg::CompressedImage;
  using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
  using DetectionArrayMsg = vision_msgs::msg::Detection2DArray;
  using BBoxCandidate = get_seg_3d_coord::BBoxCandidate;
  using ClusterResult = get_seg_3d_coord::ClusterResult;
  using PointSelectionConfig = get_seg_3d_coord::PointSelectionConfig;
  using PointSelectionStats = get_seg_3d_coord::PointSelectionStats;
  using StateOdometry = get_seg_3d_coord::StateOdometry;
  using SyncedInput = get_seg_3d_coord::SyncedInput;
  using InputSyncPolicy =
      message_filters::sync_policies::ApproximateTime<MaskMsg,
                                                       DetectionArrayMsg>;
  using InputSynchronizer = message_filters::Synchronizer<InputSyncPolicy>;

  rclcpp_action::GoalResponse
  handleGoal(const rclcpp_action::GoalUUID &,
             std::shared_ptr<const LocalSearch::Goal> goal) {
    if (goal->class_names.empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "Rejecting goal: class_names is empty.");
      return rclcpp_action::GoalResponse::REJECT;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    if (goal_running_) {
      RCLCPP_WARN(this->get_logger(),
                  "Rejecting goal because another goal is already running.");
      return rclcpp_action::GoalResponse::REJECT;
    }
    goal_running_ = true;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse
  handleCancel(const std::shared_ptr<GoalHandle>) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      goal_running_ = false;
    }
    data_cv_.notify_all();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(const std::shared_ptr<GoalHandle> goal_handle) {
    std::thread{std::bind(&LocalSearchActionServer::executeAction, this,
                          std::placeholders::_1),
                goal_handle}
        .detach();
  }

  void executeAction(const std::shared_ptr<GoalHandle> goal_handle) {
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<LocalSearch::Result>();
    auto feedback = std::make_shared<LocalSearch::Feedback>();

    feedback->state = "WAIT_INPUTS";
    feedback->current_num = 0;
    goal_handle->publish_feedback(feedback);

    const auto requested_classes = std::vector<std::string>(
        goal->class_names.begin(), goal->class_names.end());
    const rclcpp::Time goal_start_stamp = this->now();

    cv::Mat mask;
    std_msgs::msg::Header mask_header;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
    DetectionArrayMsg::ConstSharedPtr bbox_msg;

    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      data_cv_.wait(
          lock,
          [this, &goal_handle, &goal_start_stamp, &requested_classes]() {
            if (goal_handle->is_canceling()) return true;
            if (!has_synced_input_ || camera_info_ == nullptr) return false;
            const auto &bm = latest_synced_input_.bbox_msg;
            if (!bm) return false;
            const rclcpp::Time bbox_stamp(bm->header.stamp);
            if (bbox_stamp < goal_start_stamp) return false;
            for (const auto &det : bm->detections) {
              if (det.results.empty()) continue;
              const auto &cid = det.results.front().hypothesis.class_id;
              if (std::find(requested_classes.begin(),
                            requested_classes.end(),
                            cid) != requested_classes.end()) {
                return true;
              }
            }
            return false;
          });

      if (goal_handle->is_canceling()) {
        goal_running_ = false;
        result->success = false;
        result->message = "canceled";
        goal_handle->canceled(result);
        return;
      }

      mask = latest_synced_input_.mask.clone();
      mask_header = latest_synced_input_.mask_header;
      camera_info = camera_info_;
      bbox_msg = latest_synced_input_.bbox_msg;
    }

    feedback->state = "WAIT_STABLE_ODOM";
    goal_handle->publish_feedback(feedback);

    const rclcpp::Time projection_stamp(mask_header.stamp);
    if (require_stable_odom_ &&
        !waitForStableOdom(goal_handle, projection_stamp)) {
      if (goal_handle->is_canceling()) {
        goal_running_ = false;
        result->success = false;
        result->message = "canceled";
        goal_handle->canceled(result);
      } else {
        abortGoal(goal_handle, result,
                  "odom was not stable around mask timestamp");
      }
      return;
    }

    feedback->state = "COLLECT_SUBMAP";
    goal_handle->publish_feedback(feedback);

    const rclcpp::Time collection_start_stamp = this->now();
    std::vector<PointCloud2Msg::ConstSharedPtr> collected_clouds;
    collected_clouds.reserve(accumulate_frames_);
    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      const auto deadline =
          std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::duration<double>(accumulate_timeout_sec_));
      while (static_cast<int>(collected_clouds.size()) < accumulate_frames_ &&
             !goal_handle->is_canceling()) {
        if (!cloud_queue_.empty()) {
          auto cm = cloud_queue_.front();
          cloud_queue_.pop_front();
          const rclcpp::Time cstamp(cm->header.stamp);
          if (cstamp >= collection_start_stamp) {
            collected_clouds.push_back(cm);
          }
          continue;
        }
        if (data_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
          break;
        }
      }
    }

    if (goal_handle->is_canceling()) {
      goal_running_ = false;
      result->success = false;
      result->message = "canceled";
      goal_handle->canceled(result);
      return;
    }

    if (collected_clouds.empty()) {
      abortGoal(goal_handle, result, "no livox frames collected within timeout");
      return;
    }

    feedback->state = "CLUSTER";
    goal_handle->publish_feedback(feedback);

    const auto candidates =
        buildBBoxCandidates(*bbox_msg, requested_classes, *camera_info, mask);

    auto accumulated_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::vector<int> per_point_bbox;
    PointSelectionStats selection_stats;
    const PointSelectionConfig selection_config = makePointSelectionConfig();
    for (const auto &cm : collected_clouds) {
      get_seg_3d_coord::collectMaskProjectedPoints(
          cm, camera_info, mask, candidates, projection_stamp,
          selection_config, tf_buffer_, this->get_logger(), *this->get_clock(),
          accumulated_cloud, per_point_bbox, &selection_stats);
    }
    accumulated_cloud->width =
        static_cast<std::uint32_t>(accumulated_cloud->points.size());
    accumulated_cloud->height = 1;
    accumulated_cloud->is_dense = false;

    std::vector<ClusterResult> clusters =
        get_seg_3d_coord::clusterPerBBox(
            accumulated_cloud, per_point_bbox, candidates, cluster_tolerance_,
            min_cluster_size_, max_cluster_size_, this->get_logger());

    std_msgs::msg::Header output_header;
    output_header.stamp = projection_stamp;
    output_header.frame_id = map_frame_;

    for (auto &c : clusters) {
      c.center_map = c.center_lidar;
    }

    publishFilteredCloud(output_header, accumulated_cloud);
    publishClusteredCloud(output_header, accumulated_cloud, clusters);
    publishCenters(output_header, clusters);
    publishMarkers(output_header, accumulated_cloud, clusters);

    const Eigen::Vector3f reference(static_cast<float>(goal->reference_position.x),
                                    static_cast<float>(goal->reference_position.y),
                                    static_cast<float>(goal->reference_position.z));

    const double effective_max_distance =
        goal->max_distance > 0.0F
            ? static_cast<double>(goal->max_distance)
            : max_distance_from_reference_m_;
    const float max_dist_sq =
        static_cast<float>(effective_max_distance * effective_max_distance);
    int found_count = 0;
    std::string per_class_diag;
    result->coords.resize(requested_classes.size());
    result->found.resize(requested_classes.size());
    for (std::size_t i = 0; i < requested_classes.size(); ++i) {
      const auto &name = requested_classes[i];
      float best_dist_sq = std::numeric_limits<float>::infinity();
      float nearest_rejected_dist =
          std::numeric_limits<float>::infinity();
      int matching_class_clusters = 0;
      const ClusterResult *best = nullptr;
      for (const auto &cluster : clusters) {
        if (cluster.class_id != name) {
          continue;
        }
        ++matching_class_clusters;
        const float d = (cluster.center_map - reference).squaredNorm();
        if (d > max_dist_sq) {
          if (d < nearest_rejected_dist) nearest_rejected_dist = d;
          continue;
        }
        if (d < best_dist_sq) {
          best_dist_sq = d;
          best = &cluster;
        }
      }
      if (best) {
        result->coords[i].x = best->center_map.x();
        result->coords[i].y = best->center_map.y();
        result->coords[i].z = best->center_map.z();
        result->found[i] = true;
        ++found_count;
      } else {
        result->coords[i].x = 0.0;
        result->coords[i].y = 0.0;
        result->coords[i].z = 0.0;
        result->found[i] = false;
        char dbuf[128];
        if (matching_class_clusters == 0) {
          std::snprintf(dbuf, sizeof(dbuf), "[%s: no cluster]", name.c_str());
        } else {
          std::snprintf(dbuf, sizeof(dbuf),
                        "[%s: %d cluster(s) but nearest %.2fm > %.2fm gate]",
                        name.c_str(), matching_class_clusters,
                        std::sqrt(nearest_rejected_dist),
                        effective_max_distance);
        }
        per_class_diag += dbuf;
      }
    }

    feedback->current_num = static_cast<std::int32_t>(found_count);
    goal_handle->publish_feedback(feedback);

    goal_running_ = false;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "found %d/%zu classes (clusters=%zu, points=%zu, frames=%zu) %s",
                  found_count, requested_classes.size(), clusters.size(),
                  accumulated_cloud->points.size(), collected_clouds.size(),
                  per_class_diag.c_str());
    result->message = buf;
    result->success = (found_count == static_cast<int>(requested_classes.size()));
    goal_handle->succeed(result);
  }

  void abortGoal(const std::shared_ptr<GoalHandle> &goal_handle,
                 const std::shared_ptr<LocalSearch::Result> &result,
                 const std::string &message) {
    goal_running_ = false;
    result->success = false;
    result->message = message;
    goal_handle->abort(result);
  }

  std::vector<BBoxCandidate>
  buildBBoxCandidates(const DetectionArrayMsg &bbox_msg,
                      const std::vector<std::string> &requested_classes,
                      const sensor_msgs::msg::CameraInfo &camera_info,
                      const cv::Mat &mask) const {
    std::vector<BBoxCandidate> candidates;
    candidates.reserve(bbox_msg.detections.size());

    const float img_w = camera_info.width > 0
                            ? static_cast<float>(camera_info.width)
                            : static_cast<float>(mask.cols);
    const float img_h = camera_info.height > 0
                            ? static_cast<float>(camera_info.height)
                            : static_cast<float>(mask.rows);

    for (const auto &det : bbox_msg.detections) {
      if (det.results.empty()) {
        continue;
      }
      const auto &top = det.results.front();
      const std::string class_id = top.hypothesis.class_id;
      if (std::find(requested_classes.begin(), requested_classes.end(),
                    class_id) == requested_classes.end()) {
        continue;
      }

      const float cx = static_cast<float>(det.bbox.center.position.x);
      const float cy = static_cast<float>(det.bbox.center.position.y);
      const float sx = static_cast<float>(det.bbox.size_x);
      const float sy = static_cast<float>(det.bbox.size_y);
      BBoxCandidate cand;
      cand.class_id = class_id;
      cand.score = static_cast<float>(top.hypothesis.score);
      cand.u_min = std::max(0.0F, cx - sx * 0.5F);
      cand.u_max = std::min(img_w - 1.0F, cx + sx * 0.5F);
      cand.v_min = std::max(0.0F, cy - sy * 0.5F);
      cand.v_max = std::min(img_h - 1.0F, cy + sy * 0.5F);
      if (cand.u_max <= cand.u_min || cand.v_max <= cand.v_min) {
        continue;
      }
      candidates.push_back(cand);
    }
    return candidates;
  }

  void syncedInputCallback(const MaskMsg::ConstSharedPtr mask_msg,
                           const DetectionArrayMsg::ConstSharedPtr bbox_msg) {
    cv::Mat mask = imageToBinaryMask(mask_msg);
    if (mask.empty()) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_synced_input_.mask = mask;
      latest_synced_input_.mask_header = mask_msg->header;
      latest_synced_input_.bbox_msg = bbox_msg;
      has_synced_input_ = true;
    }
    data_cv_.notify_all();
  }

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      camera_info_ = msg;
    }
    data_cv_.notify_all();
  }

  void cloudCallback(const PointCloud2Msg::ConstSharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      cloud_queue_.push_back(msg);
      while (cloud_queue_.size() >
             static_cast<std::size_t>(cloud_queue_max_size_)) {
        cloud_queue_.pop_front();
      }
    }
    data_cv_.notify_all();
  }

  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      state_odometry_.push(msg);
    }
    data_cv_.notify_all();
  }

  bool waitForStableOdom(const std::shared_ptr<GoalHandle> &goal_handle,
                         const rclcpp::Time &stamp) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(odom_stable_timeout_sec_));
    while (!goal_handle->is_canceling()) {
      std::string reason;
      if (state_odometry_.isStableAround(
              stamp, odom_stable_window_sec_,
              max_relative_angular_velocity_rad_s_, &reason)) {
        return true;
      }
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Waiting stable odom near mask stamp: %s",
                            reason.c_str());
      if (data_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        RCLCPP_WARN(this->get_logger(),
                    "Odom was not stable around mask stamp within %.2fs: %s",
                    odom_stable_timeout_sec_, reason.c_str());
        return false;
      }
    }
    return false;
  }

  PointSelectionConfig makePointSelectionConfig() const {
    PointSelectionConfig config;
    config.camera_frame = camera_frame_param_;
    config.output_frame = map_frame_;
    config.tf_timeout_sec = tf_timeout_sec_;
    config.prefilter_cloud = prefilter_cloud_;
    config.prefilter_frame = prefilter_frame_;
    config.prefilter_x_min = prefilter_x_min_;
    config.prefilter_x_max = prefilter_x_max_;
    config.prefilter_y_min = prefilter_y_min_;
    config.prefilter_y_max = prefilter_y_max_;
    config.prefilter_z_min = prefilter_z_min_;
    config.prefilter_z_max = prefilter_z_max_;
    return config;
  }

  cv::Mat imageToBinaryMask(const MaskMsg::ConstSharedPtr &mask_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(mask_msg);
    } catch (const cv_bridge::Exception &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Failed to convert mask image: %s", ex.what());
      return {};
    }

    cv::Mat gray;
    const cv::Mat &src = cv_ptr->image;
    if (src.channels() == 1) {
      gray = src;
    } else if (src.channels() == 3) {
      cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
      cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Unsupported mask channel count: %d",
                           src.channels());
      return {};
    }

    cv::Mat gray_u8;
    if (gray.depth() == CV_8U) {
      gray_u8 = gray;
    } else {
      gray.convertTo(gray_u8, CV_8U);
    }

    cv::Mat binary;
    cv::threshold(gray_u8, binary, mask_threshold_, 255, cv::THRESH_BINARY);
    return binary;
  }

  void publishFilteredCloud(
      const std_msgs::msg::Header &header,
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud) const {
    PointCloud2Msg output;
    pcl::toROSMsg(*cloud, output);
    output.header = header;
    filtered_cloud_pub_->publish(output);
  }

  void publishCenters(const std_msgs::msg::Header &header,
                      const std::vector<ClusterResult> &clusters) const {
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header = header;
    for (const auto &c : clusters) {
      geometry_msgs::msg::Pose pose;
      pose.position.x = c.center_lidar.x();
      pose.position.y = c.center_lidar.y();
      pose.position.z = c.center_lidar.z();
      pose.orientation.w = 1.0;
      pose_array.poses.push_back(pose);
    }
    centers_pub_->publish(pose_array);
  }

  void publishClusteredCloud(
      const std_msgs::msg::Header &header,
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
      const std::vector<ClusterResult> &clusters) const {
    std::size_t total_points = 0;
    for (const auto &c : clusters) {
      total_points += c.indices.indices.size();
    }

    sensor_msgs::msg::PointCloud2 output;
    output.header = header;
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2Fields(
        4, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
        sensor_msgs::msg::PointField::FLOAT32, "z", 1,
        sensor_msgs::msg::PointField::FLOAT32, "cluster_id", 1,
        sensor_msgs::msg::PointField::INT32);
    modifier.resize(total_points);

    sensor_msgs::PointCloud2Iterator<float> iter_x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(output, "z");
    sensor_msgs::PointCloud2Iterator<std::int32_t> iter_cluster_id(
        output, "cluster_id");

    for (std::size_t k = 0; k < clusters.size(); ++k) {
      for (const int point_idx : clusters[k].indices.indices) {
        const auto &point = cloud->points[point_idx];
        *iter_x = point.x;
        *iter_y = point.y;
        *iter_z = point.z;
        *iter_cluster_id = static_cast<std::int32_t>(k);
        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_cluster_id;
      }
    }

    clustered_cloud_pub_->publish(output);
  }

  void publishMarkers(
      const std_msgs::msg::Header &header,
      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
      const std::vector<ClusterResult> &clusters) const {
    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header = header;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    for (std::size_t i = 0; i < clusters.size(); ++i) {
      const auto &cr = clusters[i];
      Eigen::Vector4f min_point;
      Eigen::Vector4f max_point;
      pcl::getMinMax3D(*cloud, cr.indices.indices, min_point, max_point);

      const double scale_x =
          std::max(static_cast<double>(max_point.x() - min_point.x()), 0.05);
      const double scale_y =
          std::max(static_cast<double>(max_point.y() - min_point.y()), 0.05);
      const double scale_z =
          std::max(static_cast<double>(max_point.z() - min_point.z()), 0.05);

      visualization_msgs::msg::Marker marker;
      marker.header = header;
      marker.ns = "local_search_clusters";
      marker.id = static_cast<int>(i);
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = cr.center_lidar.x();
      marker.pose.position.y = cr.center_lidar.y();
      marker.pose.position.z = cr.center_lidar.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = scale_x;
      marker.scale.y = scale_y;
      marker.scale.z = scale_z;
      marker.color.r = 0.1F;
      marker.color.g = 0.8F;
      marker.color.b = 1.0F;
      marker.color.a = 0.35F;
      marker_array.markers.push_back(marker);

      visualization_msgs::msg::Marker text_marker;
      text_marker.header = header;
      text_marker.ns = "local_search_cluster_labels";
      text_marker.id = static_cast<int>(i);
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose.position.x = cr.center_lidar.x();
      text_marker.pose.position.y = cr.center_lidar.y();
      text_marker.pose.position.z = cr.center_lidar.z() + scale_z * 0.5 + 0.15;
      text_marker.pose.orientation.w = 1.0;
      text_marker.scale.z = 0.18;
      text_marker.color.r = 1.0F;
      text_marker.color.g = 1.0F;
      text_marker.color.b = 1.0F;
      text_marker.color.a = 1.0F;
      char score_buf[16];
      std::snprintf(score_buf, sizeof(score_buf), "%.2f", cr.score);
      text_marker.text = cr.class_id + " (" + score_buf + ") n=" +
                         std::to_string(cr.indices.indices.size());
      marker_array.markers.push_back(text_marker);
    }

    markers_pub_->publish(marker_array);
  }

  std::string mask_topic_;
  std::string pointcloud_topic_;
  std::string camera_info_topic_;
  std::string bbox_topic_;
  std::string odom_topic_;
  std::string filtered_cloud_topic_;
  std::string centers_topic_;
  std::string clustered_cloud_topic_;
  std::string markers_topic_;
  std::string action_name_;
  std::string camera_frame_param_;
  std::string map_frame_;
  int mask_threshold_{0};
  double cluster_tolerance_{0.30};
  int min_cluster_size_{8};
  int max_cluster_size_{30000};
  int input_sync_queue_size_{10};
  double input_sync_slop_sec_{0.2};
  double tf_timeout_sec_{0.05};
  bool require_stable_odom_{false};
  double odom_stable_window_sec_{0.3};
  double odom_stable_timeout_sec_{1.0};
  double max_relative_angular_velocity_rad_s_{0.20};
  int odom_buffer_size_{200};
  bool prefilter_cloud_{false};
  std::string prefilter_frame_{"base"};
  double prefilter_x_min_{0.0};
  double prefilter_x_max_{5.0};
  double prefilter_y_min_{-2.0};
  double prefilter_y_max_{2.0};
  double prefilter_z_min_{-0.05};
  double prefilter_z_max_{2.0};
  int accumulate_frames_{10};
  double accumulate_timeout_sec_{3.0};
  int cloud_queue_max_size_{60};
  double max_distance_from_reference_m_{1.0};

  std::mutex data_mutex_;
  std::condition_variable data_cv_;
  bool has_synced_input_{false};
  bool goal_running_{false};
  SyncedInput latest_synced_input_;
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  std::deque<PointCloud2Msg::ConstSharedPtr> cloud_queue_;
  StateOdometry state_odometry_;

  message_filters::Subscriber<MaskMsg> mask_sub_;
  message_filters::Subscriber<DetectionArrayMsg> bbox_sub_;
  std::shared_ptr<InputSynchronizer> input_sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<PointCloud2Msg>::SharedPtr filtered_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr centers_pub_;
  rclcpp::Publisher<PointCloud2Msg>::SharedPtr clustered_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      markers_pub_;
  rclcpp_action::Server<LocalSearch>::SharedPtr action_server_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalSearchActionServer>());
  rclcpp::shutdown();
  return 0;
}
