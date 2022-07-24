#ifndef HAND_LOCALIZATION
#define HAND_LOCALIZATION

#include <opencv2/core.hpp>

std::vector<std::vector<cv::Rect>> localizeHands(std::string datasetPath, std::string weightsPath, std::vector<std::string> imgsPath);

void runYoloDetection(std::string datasetPath, std::string weightsPath);
std::string getLabelsName(std::string imageFilename);
std::vector<cv::Rect> loadFromYolo(std::string yoloLabelsPath, double imgHeight, double imgWidth);



std::vector<cv::Rect> localizeHands_opencvNN(cv::Mat &img);
cv::Mat letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride);
void convertLetterboxCoords(cv::Rect &bbox, cv::Mat &img);



void saveLabels2PtNotation(std::vector<std::string> imgsPath, std::string saveDir, std::vector<std::vector<cv::Rect>> allDatasetBBoxes);

void showBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes, int imgNum);     // show bounding boxes

#endif // HAND_LOCALIZATION