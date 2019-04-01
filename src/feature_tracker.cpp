#include "feature_tracker/feature_tracker.h"
#include "feature_tracker/utils.h"


namespace tracker
{


FeatureTracker::FeatureTracker() {}
FeatureTracker::~FeatureTracker() {}


FeatureTracker::FeatureTracker(const std::string& filename)
{
  load(filename);
}


void FeatureTracker::run(const cv::Mat& image)
{
  if (dist_mask_.empty())
    createDistortionMask(image.size());

  trackFeatures(image);
}


void FeatureTracker::load(const std::string& filename)
{
  int klt_win_size, klt_iters;
  double klt_max_error;
  get_yaml_node("show_image", filename, show_image_);
  get_yaml_node("image_wait_time", filename, image_wait_);
  get_yaml_node("max_tracked_features", filename, num_features_);
  get_yaml_node("min_feature_quality", filename, min_feat_quality_);
  get_yaml_node("min_feature_distance", filename, min_feature_distance_);
  get_yaml_node("feat_filter_disparity", filename, track_disparity_);
  get_yaml_node("klt_window_size", filename, klt_win_size);
  get_yaml_node("klt_pyramid_levels", filename, klt_max_level_);
  get_yaml_node("klt_max_iterations", filename, klt_iters);
  get_yaml_node("klt_max_error", filename, klt_max_error);
  get_yaml_node("fundamental_matrix_probability", filename, F_prob_);
  get_yaml_node("fundamental_matrix_threshold", filename, F_thresh_);

  klt_win_size_ = cv::Size(klt_win_size,klt_win_size);
  klt_term_crit_ = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_iters, klt_max_error);
  next_feature_id_ = 0;
  next_image_id_ = -1;

  detector_ = cv::FastFeatureDetector::create(min_feat_quality_);

  if (show_image_)
    cv::namedWindow(OPENCV_WINDOW);

  double fx, fy, cx, cy, k1, k2, k3, p1, p2;
  get_yaml_node("fx", filename, fx);
  get_yaml_node("fy", filename, fy);
  get_yaml_node("cx", filename, cx);
  get_yaml_node("cy", filename, cy);
  get_yaml_node("k1", filename, k1);
  get_yaml_node("k2", filename, k2);
  get_yaml_node("k3", filename, k3);
  get_yaml_node("p1", filename, p1);
  get_yaml_node("p2", filename, p2);

  // camera_matrix_ (K) is 3x3
  camera_matrix_ = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));
  camera_matrix_.at<double>(0, 0) = fx;
  camera_matrix_.at<double>(1, 1) = fy;
  camera_matrix_.at<double>(0, 2) = cx;
  camera_matrix_.at<double>(1, 2) = cy;
  camera_matrix_.at<double>(2, 2) = 1.0;

  // dist_coeff_    (D) is a column vector of 4, 5, or 8 elements (only 5 in this code)
  dist_coeff_ = cv::Mat(5, 1, CV_64FC1, cv::Scalar(0));
  dist_coeff_.at<double>(0, 0) = k1;
  dist_coeff_.at<double>(1, 0) = k2;
  dist_coeff_.at<double>(2, 0) = p1;
  dist_coeff_.at<double>(3, 0) = p2;
  dist_coeff_.at<double>(4, 0) = k3;
}


// Taken from visual_mtt2 and modified
void FeatureTracker::createDistortionMask(cv::Size res)
{
  // Define undistorted image boundary
  int num_ppe = 10; // number of points per edge
  std::vector<cv::Point2f> boundary;
  for (uint32_t i = 0; i < num_ppe; i++)
    boundary.emplace_back(cv::Point2f(i*(res.width/num_ppe), 0)); // bottom
  for (uint32_t i = 0; i < num_ppe; i++)
    boundary.emplace_back(cv::Point2f(res.width, i*(res.height/num_ppe))); // right
  for (uint32_t i = 0; i < num_ppe; i++)
    boundary.emplace_back(cv::Point2f(res.width - i*(res.width/num_ppe), res.height)); // top
  for (uint32_t i = 0; i < num_ppe; i++)
    boundary.emplace_back(cv::Point2f(0, res.height - i*(res.height/num_ppe))); // left

  // Project points onto the normalized image plane
  cv::Mat dist_coeff; // we started with the theoretical undistorted image
  cv::undistortPoints(boundary, boundary, camera_matrix_, dist_coeff);

  // Put points into homogeneous coordinates and project onto the image frame
  std::vector<cv::Point3f> boundary_h; // homogeneous
  std::vector<cv::Point2f> boundary_d; // distorted
  cv::convertPointsToHomogeneous(boundary, boundary_h);
  cv::projectPoints(boundary_h, cv::Vec3f(0,0,0), cv::Vec3f(0,0,0), camera_matrix_, dist_coeff_, boundary_d);

  // Convert boundary to mat and create the mask by filling in a polygon defined by the boundary
  cv::Mat boundary_mat(boundary_d);
  boundary_mat.convertTo(boundary_mat, CV_32SC1);
  dist_mask_ = cv::Mat(res, CV_8UC1, cv::Scalar(0));
  cv::fillConvexPoly(dist_mask_, boundary_mat, cv::Scalar(1));

  // Save boundary points for drawing boundary line
  cv::Mat canny_output;
  cv::Canny(dist_mask_, canny_output, 0, 2, 3);
  cv::findContours(canny_output, contours_, hierarchy_, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
}


bool FeatureTracker::getNewFeatures(cv::Mat img)
{
  // create mask around matched features with filled circles
  cv::Mat mask;
  dist_mask_.copyTo(mask);
  static cv::Scalar color = 0;
  static int thickness = -1; // negative means filled circle
  for (auto& f : features_)
    cv::circle(mask, f, min_feature_distance_, color, thickness);

  // collect new features outside the mask
  std::vector<cv::KeyPoint> new_keypoints;
  detector_->detect(img, new_keypoints, mask);
  if (new_keypoints.empty())
  {
    std::cout << "Feature detector is unable to find features!\n";
    features_.clear();
    feature_ids_.clear();
    return false;
  }

  // keep best points with good separation from others
  auto good_keypoints = chooseKeyPoints(new_keypoints, img.cols, img.rows);

  // convert keypoints to point2f and assing labels
  std::vector<cv::Point2f> new_features;
  std::vector<int> new_feature_ids;
  cv::KeyPoint::convert(good_keypoints, new_features);
  for (int i = next_feature_id_; i < (next_feature_id_ + new_features.size()); i++)
    new_feature_ids.push_back(i);

  // increment the next feature id number
  next_feature_id_ += new_features.size();

  // add new features and corresponding labels into their respective containers
  for (int i = 0; i < new_features.size(); ++i)
  {
    features_.push_back(new_features[i]);
    feature_ids_.push_back(new_feature_ids[i]);
  }

  return true;
}


void FeatureTracker::trackerInit(cv::Mat img)
{
  if (!getNewFeatures(img)) return;
  img_kf_ = img.clone();
  features_kf_ = features_;
  feature_ids_kf_ = feature_ids_;
  img.copyTo(img_prev_);
  if (show_image_)
    plotFeatures(img.clone(), features_, feature_ids_);
}


void FeatureTracker::trackFeatures(cv::Mat img)
{
  // When no features are currently tracked, detect features and establish keyframe
  if (features_.empty())
  {
    trackerInit(img);
    return;
  }

  // calculate the optical flow
  std::vector<cv::Point2f> new_features;
  std::vector<uchar> flow_inlier;
  cv::calcOpticalFlowPyrLK(img_prev_, img, features_, new_features, flow_inlier, cv::noArray(), klt_win_size_, klt_max_level_, klt_term_crit_);

  // flags keep features inside of the undistortable mask
  std::vector<uchar> dist_inlier;
  for (auto& f : features_)
    dist_inlier.push_back(dist_mask_.at<uchar>(int(f.y), int(f.x)));

  // collect the matched features and their indices
  std::vector<cv::Point2f> klt_matches;
  std::vector<int> klt_match_ids;
  for (int i = 0; i < flow_inlier.size(); i++)
  {
    if (flow_inlier[i] && dist_inlier[i])
    {
      klt_matches.push_back(new_features[i]);
      klt_match_ids.push_back(feature_ids_[i]);
    }
  }

  // restart tracker if too many features are lost
  if (klt_matches.size() < 8)
  {
    std::cout << "Too few features to calculate Fundamental matrix. Reinitializing tracker!\n";
    trackerInit(img);
    return;
  }

  // collect distances and points of matching features from keyframe
  std::vector<float> distances;
  std::vector<cv::Point2f> matches_cf, matches_kf; // current frame and keyframe matched points
  std::vector<int> match_ids;
  for (int i = 0; i < klt_match_ids.size(); i++)
  {
    for (int j = 0; j < feature_ids_kf_.size(); j++)
    {
      if (klt_match_ids[i] == feature_ids_kf_[j])
      {
        matches_cf.push_back(klt_matches[i]);
        matches_kf.push_back(features_kf_[j]);
        match_ids.push_back(klt_match_ids[i]);
        distances.push_back(cv::norm(klt_matches[i] - features_kf_[j]));
        break;
      }
    }
  }

  // compute median movement
  std::sort(distances.begin(), distances.end());
  float movement = distances[int(distances.size()/2)];

  // detect features and establish new keyframe
  if (movement > track_disparity_)
  {
    cv::Mat F_inliers;
    std::vector<cv::Point2f> matches_cf_ud, matches_kf_ud;
    cv::undistortPoints(matches_cf, matches_cf_ud, camera_matrix_, dist_coeff_, cv::noArray(), camera_matrix_);
    cv::undistortPoints(matches_kf, matches_kf_ud, camera_matrix_, dist_coeff_, cv::noArray(), camera_matrix_);
    cv::Mat F = cv::findFundamentalMat(matches_kf_ud, matches_cf_ud, cv::FM_RANSAC, F_thresh_, F_prob_, F_inliers);

    // remove outliers from feature set
    features_.clear();
    feature_ids_.clear();
    for (int i = 0; i < F_inliers.rows; i++)
    {
      if (F_inliers.at<bool>(i))
      {
        features_.push_back(matches_cf[i]);
        feature_ids_.push_back(match_ids[i]);
      }
    }

    if (features_.size() < num_features_)
      getNewFeatures(img);

    // establish keyframe feature set
    img_kf_ = img.clone();
    features_kf_ = features_;
    feature_ids_kf_ = feature_ids_;
  }
  else
  {
    features_.swap(klt_matches);
    feature_ids_.swap(klt_match_ids);
  }

  // save current image for next iteration
  img.copyTo(img_prev_);

  if (show_image_)
    plotFeatures(img.clone(), features_, feature_ids_);
}


void FeatureTracker::plotFeatures(cv::Mat img, std::vector<cv::Point2f> features, std::vector<int> features_idx)
{
  // convert image to color
  cv::cvtColor(img, img, cv::COLOR_GRAY2BGR, 3);

  // draw feature locations and corresponding ids
  for (int i = 0; i < features.size(); i++)
  {
    cv::circle(img, features[i], 5, cv::Scalar(0,255,0), 1);
    cv::putText(img, std::to_string(features_idx[i]), features[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255));
  }

  // draw undistortable boundary
  for(int i = 0; i < contours_.size(); i++)
    cv::drawContours(img, contours_, i, cv::Scalar(255,255,0), 1, 8, hierarchy_, 0, cv::Point());

  cv::imshow(OPENCV_WINDOW, img);
  cv::waitKey(image_wait_);
}


std::vector<cv::KeyPoint> FeatureTracker::chooseKeyPoints(std::vector<cv::KeyPoint> keypoints, const int& image_width, const int& image_height)
{
  std::vector<cv::KeyPoint> good_keypoints;
  size_t i, j, num_keypoints = keypoints.size();

  // Keep only the best of the number needed
  cv::KeyPointsFilter::retainBest(keypoints, num_features_ - features_.size());

  // Don't bother checking distance unless minimum distance is significant
  if (min_feature_distance_ >= 1)
  {
    // Partition image into a grid
    const int cell_size = cvRound(min_feature_distance_);
    const int grid_width = (image_width + cell_size - 1) / cell_size;
    const int grid_height = (image_height + cell_size - 1) / cell_size;
    std::vector<std::vector<cv::Point2f> > grid(grid_width*grid_height);

    // Compute squared minimum distance for comparison later
    static double md2 = min_feature_distance_ * min_feature_distance_;

    // Loop through keypoints, keeping better ones first
    for(i = 0; i < num_keypoints; i++)
    {
      // Get feature point components
      float x = keypoints[i].pt.x;
      float y = keypoints[i].pt.y;

      // Assume good point unless proven otherwise
      bool good = true;

      // Determine points position in the grid
      int x_cell = (int)x / cell_size;
      int y_cell = (int)y / cell_size;

      // Define boundary cells
      int x1 = x_cell - 1;
      int y1 = y_cell - 1;
      int x2 = x_cell + 1;
      int y2 = y_cell + 1;

      // Prevent overstepping at boundaries
      x1 = std::max(0, x1);
      y1 = std::max(0, y1);
      x2 = std::min(grid_width-1, x2);
      y2 = std::min(grid_height-1, y2);

      // Check distance from points in own and surrounding grid cells
      for(int yy = y1; yy <= y2; yy++)
      {
        for(int xx = x1; xx <= x2; xx++)
        {
          // Pull out points from current grid cell
          std::vector<cv::Point2f> m = grid[yy*grid_width + xx];

          // Check distance from points in current cell
          if(!m.empty())
          {
            for(j = 0; j < m.size(); j++)
            {
              float dx = x - m[j].x;
              float dy = y - m[j].y;

              // Drop keypoint if it's too close to another one
              if(dx*dx + dy*dy < md2)
              {
                good = false;
                goto break_out;
              }
            }
          }
        }
      }

      break_out:

      // If keypoint is not too close to another one, add it to the grid and save it
      if (good)
      {
        grid[y_cell*grid_width + x_cell].emplace_back(cv::Point2f(x, y));
        good_keypoints.push_back(keypoints[i]);
      }
    }
  }
  else
  {
    // Since minimum distance wasn't significant, keep all keypoints
    for(i = 0; i < num_keypoints; i++)
      good_keypoints.push_back(keypoints[i]);
  }

  return good_keypoints;
}


} // namespace tracker
