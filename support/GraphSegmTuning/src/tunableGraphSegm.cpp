#include <tunableGraphSegm.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

using namespace std;
using namespace cv;

void applyFilterOnSliders(int val, void *userdata)
{
    tunableGraphSegm s = *(tunableGraphSegm*)userdata;

    s.sigma = ((double)s.intSigma)/10.F;
    s.k = ((float)s.intK);

    Ptr<ximgproc::segmentation::GraphSegmentation> segmentor = ximgproc::segmentation::createGraphSegmentation(s.sigma, s.k, s.minSize);

    Mat mask = Mat::zeros(s.original.size(), CV_32SC1);
    segmentor.get()->processImage(s.original, mask);

    // Find maximum
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);

    // generate color for each region
    vector<Vec3b> regionsColor;
    regionsColor.resize((int)maxVal);
    for (int i = 0; i < regionsColor.size(); i++)
        regionsColor[i] = Vec3b((uchar)theRNG().uniform(0,255), (uchar)theRNG().uniform(0,255), (uchar)theRNG().uniform(0,255));
    

    // draw segmentation
    Mat segmentedOutput = Mat::zeros(s.original.size(), CV_8UC3);
    
    int regionValue;
    for (int row = 0; row < mask.rows; row++)
    {
        for (int col = 0; col < mask.cols; col++)
        {
            regionValue = mask.at<int>(row,col);
            segmentedOutput.at<Vec3b>(row,col) = regionsColor[regionValue];
        }
    }
    
    // scale image
    Mat scaled;
    resize(segmentedOutput, scaled, Size(s.original.cols*3, s.original.rows*3));

    int centerX = scaled.cols/2, centerY = scaled.rows/2;
    int w = scaled.cols / 10, h = scaled.rows / 10;
    rectangle(scaled, Rect(centerX, centerY, w, h), Scalar(0,0,255), 1, LINE_4);

    imshow(s.windowName, scaled);
}

void viewDemoRoulette(vector<Mat> imgs)
{
    tunableGraphSegm s;
    s.original = imgs[0];

    s.intK = 100;
    s.intSigma = 5;

    s.sigma = 0.5F;
    s.k = 80.;
    s.minSize = 200;

    namedWindow("original");
    namedWindow(s.windowName);
    createTrackbar("min size", s.windowName, &s.minSize, 400, applyFilterOnSliders, &s);
    createTrackbar("k", s.windowName, &s.intK, 300, applyFilterOnSliders, &s);
    createTrackbar("Sigma * 10", s.windowName, &s.intSigma, 20, applyFilterOnSliders, &s);

    char nxt = 't';
    int i = 0;
    
    printCommands();

    do
    {
        if ((nxt == 'd') && (i < imgs.size()-1))
            i++;
        else if ((nxt == 'a') && (i > 0))
            i--;
        else if ((nxt == 'q') || (nxt == 27))   // quit with 'q' or ESC key
            break;

        s.original = imgs[i];

        Mat scaled;
        resize(s.original, scaled, Size(s.original.cols*2, s.original.rows*2));

        int centerX = scaled.cols/2, centerY = scaled.rows/2;
        int w = scaled.cols / 8, h = scaled.rows / 8;
        rectangle(scaled, Rect(centerX, centerY, w, h), Scalar(0,0,255), 1, LINE_4);

        imshow("original", scaled);
        setWindowTitle("original", "Original Image " + to_string(i));

        applyFilterOnSliders(0, &s);
        setWindowTitle(s.windowName, "Graph Segmentation Regions " + to_string(i));

    } while (nxt = (char)waitKey(0));
}

void printCommands()
{
    cout << "Press 'q' or ESC key to quit (or exit current presentation)" << endl;
    cout << "Press 'd' to move to next picture "  << endl;
    cout << "Press 'a' to move to previous picture "  << endl;
}