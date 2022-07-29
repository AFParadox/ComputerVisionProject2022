#ifndef TUNABLE_BILATERAL
#define TUNABLE_BILATERAL

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

struct tunableGraphSegm
{
    double sigma;
    float k;
    int minSize, intSigma, intK;
    cv::Mat original;
    const char *windowName = "graphSegm";
};

void viewDemoRoulette(std::vector<cv::Mat> imgs);
void applyFilterOnSliders(int val, void *userdata);

void printCommands();

#endif //TUNABLE_BILATERAL