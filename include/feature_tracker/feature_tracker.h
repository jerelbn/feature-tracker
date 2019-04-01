#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "feature_tracker/feature_tracker.h"

static const std::string OPENCV_WINDOW = "Image Window";


namespace tracker
{


class FeatureTracker
{
public:

  FeatureTracker ();
  FeatureTracker(const std::string& filename);
  ~FeatureTracker();

  void load(const std::string& filename);
  void run(const cv::Mat& image);

private:

  bool show_image_;
  double min_feature_distance_;
  cv::Mat camera_matrix_, dist_coeff_, dist_mask_;
  cv::Ptr<cv::Feature2D> detector_;
  std::vector<std::vector<cv::Point> > contours_; // for drawing distortion mask
  std::vector<cv::Vec4i> hierarchy_; // for drawing distortion mask

  double track_disparity_;
  double F_prob_;
  double F_thresh_;
  double image_wait_;
  int num_features_;
  int min_feat_quality_;
  int next_feature_id_;
  int next_image_id_;
  int klt_max_level_;
  std::vector<cv::Point2f> features_;
  std::vector<int> feature_ids_;
  cv::Mat img_prev_;
  cv::Mat img_kf_;
  cv::Size klt_win_size_;
  cv::TermCriteria klt_term_crit_;
  std::vector<cv::Point2f> features_kf_;
  std::vector<int> feature_ids_kf_;

  void trackFeatures(cv::Mat img);
  void trackerInit(cv::Mat img);
  void createDistortionMask(cv::Size res);
  bool getNewFeatures(cv::Mat img);
  void plotFeatures(cv::Mat img, std::vector<cv::Point2f> features, std::vector<int> features_idx);
  std::vector<cv::KeyPoint> chooseKeyPoints(std::vector<cv::KeyPoint> keypoints, const int& image_width, const int& image_height);
};


} // namespace tracker
