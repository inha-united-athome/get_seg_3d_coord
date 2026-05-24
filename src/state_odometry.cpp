#include "get_seg_3d_coord/state_odometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

using nav_msgs::msg::Odometry;

namespace get_seg_3d_coord {

double calculateRelativeAngularVelocity(const Odometry &odom1,
                                        const Odometry &odom2) {
  Eigen::Quaterniond q1(odom1.pose.pose.orientation.w,
                        odom1.pose.pose.orientation.x,
                        odom1.pose.pose.orientation.y,
                        odom1.pose.pose.orientation.z);
  Eigen::Quaterniond q2(odom2.pose.pose.orientation.w,
                        odom2.pose.pose.orientation.x,
                        odom2.pose.pose.orientation.y,
                        odom2.pose.pose.orientation.z);
  q1.normalize();
  q2.normalize();

  Eigen::Quaterniond q_relative = q1.conjugate() * q2;
  q_relative.normalize();
  const double angle =
      2.0 * std::atan2(q_relative.vec().norm(), std::abs(q_relative.w()));
  const double time_diff =
      (odom2.header.stamp.sec - odom1.header.stamp.sec) +
      (odom2.header.stamp.nanosec - odom1.header.stamp.nanosec) * 1e-9;

  if (time_diff <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs(angle) / time_diff;
}

void StateOdometry::push(const Odometry::ConstSharedPtr &odom) {
  if (!odom) {
    return;
  }
  odom_queue_.push_back(*odom);
  while (odom_queue_.size() > max_buffer_size_) {
    odom_queue_.pop_front();
  }
}

bool StateOdometry::isStableAround(
    const rclcpp::Time &stamp,
    const double window_sec,
    const double max_relative_angular_velocity,
    std::string *reason) const {
  if (odom_queue_.size() < 2) {
    if (reason) *reason = "not enough odom samples";
    return false;
  }

  const rclcpp::Time start = stamp - rclcpp::Duration::from_seconds(window_sec);
  std::vector<Odometry> window;
  for (const auto &odom : odom_queue_) {
    const rclcpp::Time odom_stamp(odom.header.stamp);
    if (odom_stamp >= start && odom_stamp <= stamp) {
      window.push_back(odom);
    }
  }

  if (window.size() < 2) {
    if (reason) *reason = "not enough odom samples in stable window";
    return false;
  }

  double max_velocity = 0.0;
  for (std::size_t i = 1; i < window.size(); ++i) {
    max_velocity = std::max(
        max_velocity,
        calculateRelativeAngularVelocity(window[i - 1], window[i]));
  }

  if (max_velocity > max_relative_angular_velocity) {
    if (reason) {
      *reason = "relative angular velocity " + std::to_string(max_velocity) +
                " > " + std::to_string(max_relative_angular_velocity);
    }
    return false;
  }
  return true;
}

std::optional<StateOdometry::Odometry> StateOdometry::latest() const {
  if (odom_queue_.empty()) {
    return std::nullopt;
  }
  return odom_queue_.back();
}

void StateOdometry::setMaxBufferSize(const std::size_t max_buffer_size) {
  max_buffer_size_ = std::max<std::size_t>(2, max_buffer_size);
  while (odom_queue_.size() > max_buffer_size_) {
    odom_queue_.pop_front();
  }
}

}  // namespace get_seg_3d_coord
