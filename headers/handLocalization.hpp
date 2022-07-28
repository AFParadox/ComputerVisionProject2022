#ifndef HAND_LOCALIZATION
#define HAND_LOCALIZATION

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>


cv::dnn::Net loadModel(std::string modelPath);  // load network model(avoid loading it each time one need to process an image)
std::vector<cv::Rect> localizeHands(cv::Mat &img, cv::dnn::Net &net);   // Run yolov5 model on one image and returns a vector of rect representing the bounding boxes(Copied and pasted from github and the adapted to our needs)
cv::Mat letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride);    // function needed to resize the image in orther to be in line with yolov5 standard(FUNCTION COPIED AND PASTED FROM GITHUB)
void convertLetterboxCoords(cv::Rect &bbox, cv::Mat &img);  // yolo-scaled image has different bounding box coordinates compared to the original one, so there is the need to conver them

double computeIOU(std::vector<cv::Rect> trueBBoxes, std::vector<cv::Rect> bboxes, int rows, int cols);

void showBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes, int imgNum);     // show bounding boxes

#endif // HAND_LOCALIZATION