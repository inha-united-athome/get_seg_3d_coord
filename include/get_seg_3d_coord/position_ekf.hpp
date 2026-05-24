#pragma once

#include <Eigen/Core>

namespace get_seg_3d_coord {

class PositionEkf {
public:
  void reset();
  bool initialized() const;
  void initialize(const Eigen::Vector3f &measurement,
                  double initial_variance);
  void predict(double dt_sec, double process_variance_per_sec);
  void update(const Eigen::Vector3f &measurement,
              double measurement_variance);
  double normalizedInnovationSquared(const Eigen::Vector3f &measurement,
                                     double measurement_variance) const;

  const Eigen::Vector3f &position() const;
  const Eigen::Matrix3f &covariance() const;
  double maxVariance() const;

private:
  bool initialized_{false};
  Eigen::Vector3f position_{Eigen::Vector3f::Zero()};
  Eigen::Matrix3f covariance_{Eigen::Matrix3f::Identity()};
};

}  // namespace get_seg_3d_coord
