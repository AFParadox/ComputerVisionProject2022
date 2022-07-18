#include <tunableBilateral.hpp>

using namespace std;
using namespace cv;

void applyFilterOnSliders(int val, void *userdata)
{
    tunableBilateral s = *(tunableBilateral*)userdata;

    if (s.kSize % 2 == 0)
        s.kSize += 1;

    s.sigmaColor = ((double)s.intSC);
    s.sigmaSpace = ((double)s.intSP);

    Mat temp = s.original.clone();

    cv::GaussianBlur(s.original, temp, cv::Size(3, 3), 100);
    cv::addWeighted(s.original, 1.5, temp, -0.5, 0, temp);

    for (int i = 0; i < s.n; i++)
    {
        bilateralFilter(temp, s.filtered, s.kSize, s.sigmaColor, s.sigmaSpace);
        temp.release();
        temp = s.filtered.clone();
    }

    imshow(s.windowName, s.filtered);
}

void viewDemoRoulette(vector<Mat> imgs)
{
    tunableBilateral s;
    s.original = imgs[0];
    s.n = 5;
    s.kSize = 3;
    s.sigmaColor = 34.;
    s.sigmaSpace = 300.;
    s.filtered = Mat::zeros(s.original.size(), s.original.type());

    namedWindow("original");
    namedWindow(s.windowName);
    createTrackbar("n", s.windowName, &s.n, 30, applyFilterOnSliders, &s);
    createTrackbar("Kernel Size", s.windowName, &s.kSize, 20, applyFilterOnSliders, &s);
    createTrackbar("Sigma Color * 10", s.windowName, &s.intSC, 600, applyFilterOnSliders, &s);
    createTrackbar("Sigma Space * 10", s.windowName, &s.intSP, 600, applyFilterOnSliders, &s);

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

        imshow("original", s.original);
        setWindowTitle("original", "Original Image " + to_string(i));

        applyFilterOnSliders(0, &s);
        setWindowTitle(s.windowName, "Bilateral Filtered Image " + to_string(i));

    } while (nxt = (char)waitKey(0));
}

void printCommands()
{
    cout << "Press 'q' or ESC key to quit (or exit current presentation)" << endl;
    cout << "Press 'd' to move to next picture "  << endl;
    cout << "Press 'a' to move to previous picture "  << endl;
}