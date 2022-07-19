#ifndef HAND_SEGMENTATION
#define HAND_SEGMENTATION

#include <opencv2/core.hpp>


struct HandData
{
    cv::Mat img;
    std::vector<cv::Rect2i> bboxes;
};

HandData loadImgAndBboxes(std::string imgPath, std::string bboxPath);





void preprocessBilateral(cv::Mat * img, int n, double sigmaColor, double sigmaSpace, int kSize);    // apply bilateral n times
void preprocessSharpenGaussian(cv::Mat * img, int kSize, double sigma);    // sharpen image by subracting its own gaussian blurred version (this was found in stack overflow)
void preprocessDrawCannyOnImg(cv::Mat * img, double t1, double t2);

cv::Mat segmentHandsWatershed(HandData data);
cv::Mat singleHandWatershed(cv::Mat hand);   // return a mask which represent single hand segmentation using watershed method
bool cmpVec3bs(cv::Vec3b v1, cv:: Vec3b v2, cv::Vec3b thresh);
void getMarkersWithGraphSegm(cv::Mat hand, cv::Rect2i centralKernel, cv::Mat * markers);



// below are functions that were useful to using while developing this part of the project
void showBBoxes(HandData data);
void saveHandIstances(std::string name, HandData data, std::string destDir);

#endif // HAND_SEGMENTATION