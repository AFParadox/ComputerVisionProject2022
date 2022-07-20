#include <handSegmentation.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>

using namespace std;
using namespace cv;

// Regulates how big the centralKernel used to define hand markers for watershed. The bigger it is the smaller the centralKernel will be
const int centralKernelParam = 10;



HandData loadImgAndBboxes(string imgPath, string bboxPath)
{
    // check if input image exists
    if (!filesystem::exists(imgPath))
    {
        cout << "File \"" << imgPath << "\" does not exists" << endl;
        exit(EXIT_FAILURE);
    }

    HandData data;

    // load img
    data.img = imread(imgPath);


    if (filesystem::exists(bboxPath))   // if labels file does not exists means that no hands were detected
    {
        // bboxes from file
        ifstream myFile(bboxPath);
        int classID;
        double propX, propY, propW, propH; // prop stands for "proportional" since yolo outputs coordinates are all divided by picure size
        int x, y, w, h;

        int j = 0;
        while (myFile >> classID >> propX >> propY >> propW >> propH)
        {
            // convert coordinates from yolo notation to classic one
            w = (int)((double)data.img.cols * propW);
            h = (int)((double)data.img.rows * propH);

            x = (int)((double)data.img.cols * propX) - w / 2;
            y = (int)((double)data.img.rows * propY) - h / 2;

            Rect2i r(x, y, w, h);
            data.bboxes.push_back(r);
            j++;
        }
    }
    

    return data;
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

void preprocessDrawCannyOnImg(cv::Mat * img, double t1, double t2)
{
    Mat cannyImg;
    Canny(*img, cannyImg, t1, t2);

    //morphologyEx(cannyImg, cannyImg, MORPH_DILATE, getStructuringElement(MORPH_CROSS, Size(3,3)), Point(-1,-1), 3);
    //morphologyEx(cannyImg, cannyImg, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(1,3)), Point(-1,-1), 1);

    for (int i = 0; i < img->rows; i++)
        for (int j = 0; j < img->cols; j++)
            if (cannyImg.at<uchar>(i,j) != 0)
                img->at<Vec3b>(i,j) = Vec3b(0,0,0);
    
}




Mat segmentHandsWatershed(HandData data)
{
    // create segmentation mask
    Mat handsMarkers = Mat::zeros(data.img.size(), CV_8U);

    // extract hand subimage and clone it in order to not preprocess areas of the original image. Do this using each hand bbox
    for (int i = 0; i < data.bboxes.size(); i++)
    {
        // crop hand and clone
        Point bboxStart(data.bboxes[i].x,data.bboxes[i].y);
        Mat original = data.img(Range(bboxStart.y, bboxStart.y + data.bboxes[i].height), Range(bboxStart.x, bboxStart.x + data.bboxes[i].width));

        Mat subhand = original.clone();

        // preprocessing
        //preprocessSharpenGaussian(&subhand, 5, 10.);
        preprocessDrawCannyOnImg(&subhand, 50., 50.);
        //preprocessBilateral(&subhand, 1, 4., 1000., 3);

        // finally apply watershed on subimage
        Mat singleHandMarkers = singleHandWatershed(original, subhand);

        // show result
        showHandPreprocSegm(original, subhand, singleHandMarkers);

        // copy submask into full size segmentation mask
        Mat subMarkers = handsMarkers(Range(bboxStart.y, bboxStart.y + data.bboxes[i].height), Range(bboxStart.x, bboxStart.x + data.bboxes[i].width));
        for (int row = 0; row < subhand.rows; row++)
            for (int col = 0; col < subhand.cols; col++)
                if (singleHandMarkers.at<int>(row,col) > 0)
                    subMarkers.at<uchar>(row,col) = (uchar)255U;
        
    }

    return handsMarkers;
}

Mat singleHandWatershed(Mat origHand, Mat preprocHand)
{
    Mat markers = Mat::zeros(origHand.size(), CV_32SC1);

    // set seed for hand class segmentation
    int xCenter = origHand.cols / 2, yCenter = origHand.rows / 2;
    int handSeedW = origHand.cols / centralKernelParam, handSeedH = origHand.rows / centralKernelParam;
    //rectangle(mask, Rect2i(xCenter - handSeedW/2, yCenter - handSeedH/2, handSeedW, handSeedH), Scalar(2), 1, LINE_4);

    // set markers for hand using graph segmentation
    setHandMarkersWithGraphSegm(preprocHand, Rect2i(xCenter - handSeedW/2, yCenter - handSeedH/2, handSeedW, handSeedH), &markers);

    // set seed for background class
    setBackgroundMarkers(origHand, &markers);

    // apply watershed
    watershed(preprocHand, markers);

    // remove background class
    subtract(markers, Mat::ones(markers.size(), markers.type()), markers);

    return markers;
}

bool cmpVec3bs(cv::Vec3b v1, cv:: Vec3b v2, cv::Vec3b thresh)
{
    for (int i = 0; i < 3; i++)
        if (abs((int)v1[i] - (int)v2[i]) <= thresh[i])
            return false;
    return true;
}

void setHandMarkersWithGraphSegm(Mat hand, Rect2i centralKernel, Mat * markers)
{
    // do graph segmentation
    Ptr<ximgproc::segmentation::GraphSegmentation> segmentor = ximgproc::segmentation::createGraphSegmentation(0.6, 100, 200);
    Mat mask;
    segmentor.get()->processImage(hand, mask);

    // iterate overcentral kernel and each region resulting of GraphSegmentation which can be found in the centralKernel is fully included in the markers map

    // determine which regions to include
    vector<int> regToInclude;
    for (int i = centralKernel.y; i < centralKernel.y + centralKernel.height; i++)
        for (int j = centralKernel.x; j < centralKernel.x + centralKernel.width; j++)
            if (find(regToInclude.begin(), regToInclude.end(), mask.at<int>(i,j)) == regToInclude.end())
                regToInclude.push_back(mask.at<int>(i,j));
    
    // include regions
    for (int i = 0; i < hand.rows; i++)
    {
        for (int j = 0; j < hand.cols; j++)
        {
            for (int l = 0; l < regToInclude.size(); l++)
            {
                if (mask.at<int>(i,j) == regToInclude[l])
                {
                    markers->at<int>(i,j) = 2;
                    break;
                }
            }
        }
    }
}

void setBackgroundMarkers(Mat img, Mat * markers)
{
    //rectangle(*markers, Rect2i(1,1, markers->cols-2, markers->rows-2), Scalar(1), 1, LINE_4);

    // now we need to remove from the background seeds those points which belong to the person arm(if wearing a t-shirt can be an issue)
    // to begin we compute mean and std dev of the center seed
    //Scalar meanScalar, stdDevScalar;

    //int xCenter = img.cols / 2, yCenter = img.rows / 2;
    //int handSeedW = img.cols / centralKernelParam, handSeedH = img.rows / centralKernelParam;
    //Mat handCenterValues = img(Range(yCenter - handSeedH/2, yCenter + handSeedH/2), Range(xCenter - handSeedW/2, xCenter + handSeedW/2));

    //meanStdDev(handCenterValues, meanScalar, stdDevScalar);

    // make mean color more lighter
    //meanScalar += Scalar(30.,30.,30.,0.);

    // set mean vector and find maximum distance from it as the L2 norm of the stdDevScalar vector
    //Vec3b mean(meanScalar[0], meanScalar[1], meanScalar[2]);
    //Vec3b stdDev(stdDevScalar[0], stdDevScalar[1], stdDevScalar[2]);

    //Vec3b mean(63, 73, 115);
    Vec3b mean(54, 87, 133);
    Vec3b stdDev(5, 5, 5);

    //double maxDistThresh = norm(stdDevScalar, NORM_L2) / 2.;
    //cout << "mean: " << mean << "   stdDev" << stdDev << endl;

    // check in each backgorund seed point if the values are between the thresholds, and, if that is the case, remove those seed points(aka arm/wrist skin)
    // done along the bounding box edges, 4 edges
    for (int i = 1; i < img.cols-1; i++)   // top edge
        if (cmpVec3bs(mean, img.at<Vec3b>(1,i), stdDev))
            markers->at<int>(1,i) = 1;
    
    for (int i = 1; i < img.cols-1; i++)   // bottom edge
        if (cmpVec3bs(mean, img.at<Vec3b>(img.rows-2,i), stdDev))
            markers->at<int>(img.rows-2,i) = 1;
    
    for (int i = 1; i < img.rows-1; i++)   // left edge
        if (cmpVec3bs(mean, img.at<Vec3b>(i,1), stdDev))
            markers->at<int>(i,1) = 1;
    
    for (int i = 1; i < img.rows-1; i++)   // right edge
        if (cmpVec3bs(mean, img.at<Vec3b>(i,img.cols-2), stdDev))
            markers->at<int>(i,img.cols-2) = 1;
}






void showBBoxes(HandData data)
{
    Mat img = data.img.clone();
    for (int i = 0; i < data.bboxes.size(); i++)
        rectangle(img, data.bboxes[i], Scalar(0,0,255), 2);
    imshow("img", img);
    waitKey(0);
}

void showHandPreprocSegm(Mat original, Mat preprocessed, Mat regionsMask)
{
    Vec3b color((uchar)0U, (uchar)0U, (uchar) 120U);
    Mat segmented = preprocessed.clone();

    for (int i = 0; i < preprocessed.rows; i++)
        for (int j = 0; j < preprocessed.cols; j++)
            if (regionsMask.at<int>(i,j) == 1)
                segmented.at<Vec3b>(i,j) = segmented.at<Vec3b>(i,j)/2 + color;
    
    // DEBUG: draw also rectangles
    int xCenter = original.cols / 2, yCenter = original.rows / 2;
    int handSeedW = original.cols / centralKernelParam, handSeedH = original.rows / centralKernelParam;

    rectangle(segmented, Rect2i(xCenter - handSeedW/2, yCenter - handSeedH/2, handSeedW, handSeedH), Scalar((uchar)0,(uchar)0,(uchar)255), 1, LINE_4);
    //rectangle(segmented, Rect2i(1,1, regionsMask.cols-2, regionsMask.rows-2), Scalar((uchar)0,(uchar)255,(uchar)0), 1, LINE_4);

    // **************************************************************************************
    //Scalar meanScalar, stdDevScalar;
    //Mat handCenterValues = original(Range(yCenter - handSeedH/2, yCenter + handSeedH/2), Range(xCenter - handSeedW/2, xCenter + handSeedW/2));

    //meanStdDev(handCenterValues, meanScalar, stdDevScalar);

    // make mean color more lighter
    //meanScalar += Scalar(30.,30.,30.,0.);

    // set mean vector and find maximum distance from it as the L2 norm of the stdDevScalar vector
    //Vec3b mean(meanScalar[0], meanScalar[1], meanScalar[2]);

    //Vec3b mean(63, 73, 115);
    Vec3b mean(54, 87, 133);
    Vec3b stdDev(5, 5, 5);

    for (int i = 1; i < original.cols-1; i++)   // top edge
        if (cmpVec3bs(mean, original.at<Vec3b>(1,i), stdDev))
            segmented.at<Vec3b>(1,i) = Vec3b((uchar)0, (uchar)255, (uchar)0);

    for (int i = 1; i < original.cols-1; i++)   // bottom edge
        if (cmpVec3bs(mean, original.at<Vec3b>(original.rows-2,i), stdDev))
            segmented.at<Vec3b>(original.rows-2,i) = Vec3b((uchar)0, (uchar)255, (uchar)0);

    for (int i = 1; i < original.rows-1; i++)   // left edge
        if (cmpVec3bs(mean, original.at<Vec3b>(i,1), stdDev))
            segmented.at<Vec3b>(i,1) = Vec3b((uchar)0, (uchar)255, (uchar)0);

    for (int i = 1; i < original.rows-1; i++)   // right edge
        if (cmpVec3bs(mean, original.at<Vec3b>(i,original.cols-2), stdDev))
            segmented.at<Vec3b>(i,original.cols-2) = Vec3b((uchar)0, (uchar)255, (uchar)0);
    // **************************************************************************************

    imshow("original", original);
    imshow("preprocessed", preprocessed);
    imshow("segmented", segmented);

    char c = (char)waitKey(0);
    if (c == 'q')
        exit(EXIT_SUCCESS);
}

void saveHandIstances(std::string name, HandData data, std::string destDir)
{
    for (int i = 0; i < data.bboxes.size(); i++)
    {
        // generate filename
        string handImgSaveLocation = destDir + name + "_" + to_string(i) + ".jpg";

        // crop hand
        Point bboxStart(data.bboxes[i].x,data.bboxes[i].y);
        Mat hand = data.img(Range(bboxStart.y, bboxStart.y + data.bboxes[i].height), Range(bboxStart.x, bboxStart.x + data.bboxes[i].width));

        // save hand
        imwrite(handImgSaveLocation, hand);
    }
}