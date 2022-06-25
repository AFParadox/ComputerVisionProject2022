#include <iostream>
#include <fstream>
#include <segmentProjectDatasetTrueRegions.hpp>

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // Check whether the input arguments are the right amount
    if (argc != 4)
        throw invalid_argument("Usage: <path>/bilTune <images pattern> <hand boxes pattern> <true masks pattern>");
    
    segmentProjectDatasetTrueRegions lib;
    int imgCount =  lib.loadImgsWithGlob(argv[1]);
    int handCount = lib.loadHandBoxesCoordsWithGlob(argv[2]);
    lib.loadTrueSegmentationMask(argv[3]);

    //lib.showImgsWithLocalizedHands();
    lib.showImgsWithTrueMasksSeg();

    return EXIT_SUCCESS;
}