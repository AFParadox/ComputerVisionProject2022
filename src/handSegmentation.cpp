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

const float GRAPH_SEGMENTATION_K = 0.4F;
const float GRAPH_SEGMENTATION_SIGMA = 50.F;
const int GRAPH_SEGMENTATION_MINSIZE = 60;


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




Mat segmentHandsWatershed(cv::Mat img, std::vector<cv::Rect> bboxes)
{
    // create segmentation mask
    Mat handsMarkers = Mat::zeros(img.size(), CV_8U);

    // extract hand subimage and clone it in order to not preprocess areas of the original image. Do this using each hand bbox
    for (int i = 0; i < bboxes.size(); i++)
    {
        // crop hand and clone
        Point bboxStart(bboxes[i].x, bboxes[i].y);
        Mat original = img(Range(bboxStart.y, bboxStart.y + bboxes[i].height), Range(bboxStart.x, bboxStart.x + bboxes[i].width));

        Mat subhand = original.clone();

        // preprocessing
        //preprocessSharpenGaussian(&subhand, 5, 10.);
        preprocessDrawCannyOnImg(&subhand, 50., 50.);
        //preprocessBilateral(&subhand, 1, 3., 1000., 3);

        // finally apply watershed on subimage
        Mat singleHandMarkers = singleHandWatershed(original, subhand);

        // show result
        //showHandPreprocSegm(original, subhand, singleHandMarkers);

        // copy submask into full size segmentation mask
        Mat subMarkers = handsMarkers(Range(bboxStart.y, bboxStart.y + bboxes[i].height), Range(bboxStart.x, bboxStart.x + bboxes[i].width));
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
    //rectangle(mask, Rect(xCenter - handSeedW/2, yCenter - handSeedH/2, handSeedW, handSeedH), Scalar(2), 1, LINE_4);

    // set markers for hand using graph segmentation
    setHandMarkersWithGraphSegm(preprocHand, Rect(xCenter - handSeedW/2, yCenter - handSeedH/2, handSeedW, handSeedH), &markers);

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


void setHandMarkersWithGraphSegm(Mat hand, Rect centralKernel, Mat * markers)
{
    // do graph segmentation
    Ptr<ximgproc::segmentation::GraphSegmentation> segmentor = ximgproc::segmentation::createGraphSegmentation(GRAPH_SEGMENTATION_SIGMA, GRAPH_SEGMENTATION_K, GRAPH_SEGMENTATION_MINSIZE);
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
    // The first line is the most basic setup of the background markers
    //rectangle(*markers, Rect(1,1, markers->cols-2, markers->rows-2), Scalar(1), 1, LINE_4);

    // This part is commented out because it's result wasn't really good
    // ************************************************************************************************************************************************************
    // Compute mean and standart deviation of central kernel region and use them to prevent portion of the hand to be included into the background markers

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

    // ************************************************************************************************************************************************************

    // mean and stdDev have these names because of the previous ad above attempt of using actual mean and stdDev.
    //Vec3b mean(63, 73, 115);
    Vec3b mean(54, 87, 133);    // this really is skin color
    Vec3b stdDev(5, 5, 5);      // this really is a threshold to use in comparisons

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


void showSegmentedHands(Mat img, Mat mask, int imgNum, Vec3b regionColor)
{
    regionColor /= 2;   // in order to not overflow the color

    // color the image on the segmented area
    Mat segmented = img.clone();
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++)
            if (mask.at<uchar>(row,col) == (uchar)255U)
                segmented.at<Vec3b>(row,col) = segmented.at<Vec3b>(row,col)/2 + regionColor;
    
    // show image with segmented hands and rename window according number(helps viewing)
    imshow("segmentationDisplay", segmented);
    setWindowTitle("segmentationDisplay", "Segmentation Result: image n° " + to_string(imgNum));
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

    rectangle(segmented, Rect(xCenter - handSeedW/2, yCenter - handSeedH/2, handSeedW, handSeedH), Scalar((uchar)0,(uchar)0,(uchar)255), 1, LINE_4);
    //rectangle(segmented, Rect(1,1, regionsMask.cols-2, regionsMask.rows-2), Scalar((uchar)0,(uchar)255,(uchar)0), 1, LINE_4);

    // This part is commented out because it's result wasn't really good
    // ************************************************************************************************************************************************************
    // Compute mean and standart deviation of central kernel region and use them to prevent portion of the hand to be included into the background markers

    //Scalar meanScalar, stdDevScalar;
    //Mat handCenterValues = original(Range(yCenter - handSeedH/2, yCenter + handSeedH/2), Range(xCenter - handSeedW/2, xCenter + handSeedW/2));

    //meanStdDev(handCenterValues, meanScalar, stdDevScalar);

    // make mean color more lighter
    //meanScalar += Scalar(30.,30.,30.,0.);

    // set mean vector and find maximum distance from it as the L2 norm of the stdDevScalar vector
    //Vec3b mean(meanScalar[0], meanScalar[1], meanScalar[2]);

    // ************************************************************************************************************************************************************

    // mean and stdDev have these names because of the previous ad above attempt of using actual mean and stdDev.
    //Vec3b mean(63, 73, 115);
    Vec3b mean(54, 87, 133);    // this really is skin color
    Vec3b stdDev(5, 5, 5);      // this really is a threshold to use in comparisons

    // move along bbox edges and draw in green background markers
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

    imshow("original", original);
    imshow("preprocessed", preprocessed);
    imshow("segmented", segmented);

    char c = (char)waitKey(0);
    if (c == 'q')
        exit(EXIT_SUCCESS);
}


void saveHandIstances(string name, Mat img, vector<Rect> bboxes, string destDir)
{
    for (int i = 0; i < bboxes.size(); i++)
    {
        // generate filename
        string handImgSaveLocation = destDir + name + "_" + to_string(i) + ".jpg";

        // crop hand
        Point bboxStart(bboxes[i].x, bboxes[i].y);
        Mat hand = img(Range(bboxStart.y, bboxStart.y + bboxes[i].height), Range(bboxStart.x, bboxStart.x + bboxes[i].width));

        // save hand
        imwrite(handImgSaveLocation, hand);
    }
}