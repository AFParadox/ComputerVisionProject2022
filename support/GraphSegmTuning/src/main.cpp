#include <iostream>
#include <fstream>
#include <tunableGraphSegm.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

vector<Mat> loadImgsFromPattern(char *pattern);

int main(int argc, char** argv)
{
    if (argc != 2)
        throw invalid_argument("usage: <exec> \"<pictures folder & pattern>\"");
    
    vector<Mat> imgs = loadImgsFromPattern(argv[1]);

    viewDemoRoulette(imgs);

    return EXIT_SUCCESS;
}

vector<Mat> loadImgsFromPattern(char *pattern)
{
    vector<string> filenames;
    glob(pattern, filenames, true);

    // check if the path is correct
    if (filenames.size() < 1)
        throw invalid_argument("There are no images inside the specified directory with the specified pattern");   

    // load imgs into vector
    vector<Mat> imgs;
    imgs.resize(filenames.size());
    for (int i = 0; i < filenames.size(); i++)
        imgs[i] = imread(filenames[i]);
    
    return imgs;
}