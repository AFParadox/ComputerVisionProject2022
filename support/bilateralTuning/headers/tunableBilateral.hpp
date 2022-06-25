#ifndef TUNABLE_BILATERAL
#define TUNABLE_BILATERAL

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

struct tunableBilateral
{
    double sigmaColor, sigmaSpace;
    int n, intSC, intSP, kSize;
    cv::Mat original, filtered;
    const char *windowName = "bilateral";
};

void viewDemoRoulette(std::vector<cv::Mat> imgs);
void applyFilterOnSliders(int val, void *userdata);

void printCommands();

#endif //TUNABLE_BILATERAL