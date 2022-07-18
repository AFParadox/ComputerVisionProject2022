#include <handSegmentation.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;

HandData loadImgAndBboxes(char * imgPath, char * bboxPath)
{
    // check if both input file exists
    if (!filesystem::exists(imgPath))
    {
        cout << "File \"" << imgPath << "\" does not exists";
        exit(EXIT_FAILURE);
    }
    if (!filesystem::exists(bboxPath))
    {
        cout << "File \"" << imgPath << "\" does not exists";
        exit(EXIT_FAILURE);
    }

    HandData data;

    // load img
    data.img = imread(imgPath);

    // bboxes from file
    ifstream myFile(bboxPath);
    int classID;
    double propX, propY, propW, propH;  // prop stands for "proportional" since yolo outputs coordinates are all divided by picure size
    int x, y, w, h;

    int j = 0;
    while (myFile >> classID >> propX >> propY >> propW >> propH)
    {
        // convert coordinates from yolo notation to classic one
        w = (int)((double)data.img.cols * propW);
        h = (int)((double)data.img.rows * propH);

        x = (int)((double)data.img.cols * propX) - w/2;
        y = (int)((double)data.img.rows * propY) - h/2;
        

        Rect2i r(x, y, w, h);
        data.bboxes.push_back(r);
        j++;
    }

    return data;
}

void showBBoxes(HandData data)
{
    Mat img = data.img.clone();
    for (int i = 0; i < data.bboxes.size(); i++)
        rectangle(img, data.bboxes[i], Scalar(0,0,255), 2);
    imshow("img", img);
    waitKey(0);
}



void preprocessBilateral(Mat * img, int n, double sigmaColor, double sigmaSpace, int kSize)
{
    // temp is needed because I don't know if using the same object both in input and output in the bilateral call is safe
    Mat temp = img->clone();

    for (int i = 0; i < n; i++)
    {
        bilateralFilter(temp, *img, kSize, sigmaColor, sigmaSpace);
        temp = img->clone();
    }
}

void preprocessSharpenGaussian(Mat *img, int kSize, double sigma)
{
    Mat gaussImg = Mat::zeros(img->size(), img->type());

    GaussianBlur(*img, gaussImg, Size(kSize, kSize), sigma);
    addWeighted(*img, 2., gaussImg, -1., 0, *img);
}

Mat singleHandWatershed(Mat hand)
{
    cout << hand.type() << endl;
    imshow("hand", hand);
    waitKey(0);


    Mat mask = Mat::zeros(hand.size(), CV_32SC1);

    // set seed points in mask
    mask.at<int>(mask.rows / 2, mask.cols / 2) = 2;

    mask.at<int>(0, 0) = 1;
    mask.at<int>(0, mask.cols - 1) = 1;
    mask.at<int>(mask.rows - 1, 0) = 1;
    mask.at<int>(mask.rows - 1, mask.cols - 1) = 1;

    // apply watershed
    watershed(hand, mask);

    // remove not useful class (2U)
    //subtract(mask, Mat::ones(mask.size(), mask.type()), mask);

    // DEBUG: display result
    Vec3b color((uchar)theRNG().uniform(0,127), (uchar)theRNG().uniform(0,127), (uchar)theRNG().uniform(0,127));
    Mat displayImg = hand.clone();
    for (int i = 0; i < hand.rows; i++)
    {
        for (int j = 0; j < hand.cols; j++)
        {
            cout << mask.at<int>(i,j) << endl;
            if (mask.at<int>(i,j) == 2)
                displayImg.at<Vec3b>(i,j) = displayImg.at<Vec3b>(i,j)/2 + color;
        }
    }
    
    imshow("segmented", displayImg);
    waitKey(0);

    return mask;
}

Mat segmentHandsWatershed(HandData data)
{
    // create segmentation mask
    Mat mask = Mat::zeros(data.img.size(), CV_8U);

    // extract hand subimage and clone it in order to not preprocess areas of the original image. Do this using each hand bbox
    for (int i = 0; i < data.bboxes.size(); i++)
    {
        // crop hand and clone
        Point bboxStart(data.bboxes[i].x,data.bboxes[i].y);
        Mat subhand = data.img(Range(bboxStart.y, bboxStart.y + data.bboxes[i].height), Range(bboxStart.x, bboxStart.x + data.bboxes[i].width)).clone();

        // preprocessing
        preprocessSharpenGaussian(&subhand, 5, 10.);
        preprocessBilateral(&subhand, 10, 10, 600, 7);

        // finally apply watershed on subimage
        Mat submask = singleHandWatershed(subhand);

        // copy submask into full size segmentation mask
    }
    return mask;
}





