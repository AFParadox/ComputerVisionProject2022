#include <handLocalization.hpp>
#include <handSegmentation.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char ** argv) 
{
    HandData data = loadImgAndBboxes(argv[1], argv[2]);
    segmentHandsWatershed(data);
}
