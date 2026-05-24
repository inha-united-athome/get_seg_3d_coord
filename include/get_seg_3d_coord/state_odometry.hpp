#pragma once

#include <deque>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace get_seg_3d_coord {

double calculateRelativeAngularVelocity(const nav_msgs::msg::Odometry &odom1,
                                        const nav_msgs::msg::Odometry &odom2);

class StateOdometry {
public:
  using Odometry = nav_msgs::msg::Odometry;

  void push(const Odometry::ConstSharedPtr &odom);
  bool isStableAround(const rclcpp::Time &stamp,
                      double window_sec,
                      double max_relative_angular_velocity,
                      std::string *reason = nullptr) const;
  std::optional<Odometry> latest() const;
  void setMaxBufferSize(std::size_t max_buffer_size);

private:
  std::deque<Odometry> odom_queue_;
  std::size_t max_buffer_size_{200};
};

}  // namespace get_seg_3d_coord
