#ifndef HAND_SEGMENTATION
#define HAND_SEGMENTATION

#include <opencv2/core.hpp>

void preprocessBilateral(cv::Mat * img, int n, double sigmaColor, double sigmaSpace, int kSize);    // apply bilateral n times
void preprocessSharpenGaussian(cv::Mat * img, int kSize, double sigma);    // sharpen image by subracting its own gaussian blurred version (this was found in stack overflow)
void preprocessDrawCannyOnImg(cv::Mat * img, double t1, double t2); // compute edges of image with canny and then draw in black them on top of the source image

cv::Mat segmentHandsWatershed(cv::Mat img, std::vector<cv::Rect> bboxes);
cv::Mat singleHandWatershed(cv::Mat origHand, cv::Mat preprocHand);   // return a mask which represent single hand segmentation using watershed method. Also include markers computation
bool cmpVec3bs(cv::Vec3b v1, cv:: Vec3b v2, cv::Vec3b thresh);  // compare two Vec3b. If they are similar within threshold returns true
void setHandMarkersWithGraphSegm(cv::Mat hand, cv::Rect centralKernel, cv::Mat * markers);    // compute hand markers using graph Segmentation and "central kernel"
void setBackgroundMarkers(cv::Mat img, cv::Mat * markers);  // compute background markers around bounding box edges while not selecting as markers pixels which color is similar to skin

void showSegmentedHands(cv::Mat img, cv::Mat mask, int imgNum, cv::Vec3b regionColor);

// below are functions that were useful to using while developing this part of the project
void showHandPreprocSegm(cv::Mat original, cv::Mat preprocessed, cv::Mat regionsMask);  // display each hand that is found in localization plus preprocessing and segmentation results
void saveHandIstances(std::string name, cv::Mat img, std::vector<cv::Rect> bboxes, std::string destDir);    // save the contents of each bounding box detected in localization task in a image

#endif // HAND_SEGMENTATION