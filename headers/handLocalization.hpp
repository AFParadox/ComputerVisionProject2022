#ifndef HAND_LOCALIZATION
#define HAND_LOCALIZATION

#include <opencv2/core.hpp>

std::vector<std::vector<cv::Rect2i>> localizeHands(std::string datasetPath, std::string weightsPath, std::vector<std::string> imgsPath);

void runYoloDetection(std::string datasetPath, std::string weightsPath);
std::string getLabelsName(std::string imageFilename);
std::vector<cv::Rect2i> loadFromYolo(std::string yoloLabelsPath, double imgHeight, double imgWidth);

void saveLabels2PtNotation(std::vector<std::string> imgsPath, std::string saveDir, std::vector<std::vector<cv::Rect2i>> allDatasetBBoxes);

void showBBoxes(cv::Mat img, std::vector<cv::Rect2i> bboxes, int imgNum);     // show bounding boxes

#endif // HAND_LOCALIZATION