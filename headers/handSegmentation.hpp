#ifndef HAND_SEGMENTATION
#define HAND_SEGMENTATION

#include <opencv2/core.hpp>


struct HandData
{
    cv::Mat img;
    std::vector<cv::Rect2i> bboxes;
};

HandData loadImgAndBboxes(char * imgPath, char * bboxPath);

void showBBoxes(HandData data);



void preprocessBilateral(cv::Mat * img, int n, double sigmaColor, double sigmaSpace, int kSize);    // apply bilateral n times
void preprocessSharpenGaussian(cv::Mat * img, int kSize, double sigma);    // sharpen image by subracting its own gaussian blurred version (this was found in stack overflow)

cv::Mat segmentHandsWatershed(HandData data);
cv::Mat singleHandWatershed(cv::Mat hand);   // return a mask which represent single hand segmentation using watershed method

#endif // HAND_SEGMENTATION