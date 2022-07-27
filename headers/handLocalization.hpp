#ifndef HAND_LOCALIZATION
#define HAND_LOCALIZATION

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>


cv::dnn::Net loadModel(std::string modelPath);
std::vector<cv::Rect> localizeHands(cv::Mat &img, cv::dnn::Net &net);
cv::Mat letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride);
void convertLetterboxCoords(cv::Rect &bbox, cv::Mat &img);

double computeIOU(std::vector<cv::Rect> trueBBoxes, std::vector<cv::Rect> bboxes, int rows, int cols);

void showBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes, int imgNum);     // show bounding boxes

#endif // HAND_LOCALIZATION