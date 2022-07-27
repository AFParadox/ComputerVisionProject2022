#include <handLocalization.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;


const float INPUT_IMG_SIZE = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

const Scalar RED = Scalar(0,0,255);
const Scalar GREY_PADDING = Scalar(114,114,114);


Net loadModel(string modelPath)
{
    // Load model
    Net model = readNet(modelPath);

    // use cuda to make computation faster
    //model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    return model;
}

vector<Rect> localizeHands(Mat &img, Net &net)     // https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python
{
    // scale and resize image with padding preventing img stretch
    Mat yoloResizedImg = letterbox(img, Size(INPUT_IMG_SIZE, INPUT_IMG_SIZE), GREY_PADDING, false, false, true, 32);

    // Convert to blob.
    Mat blob;
    blobFromImage(yoloResizedImg, blob, 1./255., Size(INPUT_IMG_SIZE, INPUT_IMG_SIZE), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> detections;
    net.forward(detections, net.getUnconnectedOutLayersNames());

    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<float> confidences;
    vector<Rect> boxes; 

    // Resizing factor.
    float x_factor = yoloResizedImg.cols / INPUT_IMG_SIZE;
    float y_factor = yoloResizedImg.rows / INPUT_IMG_SIZE;

    float *data = (float *)detections[0].data;

    const int dimensions = 6;   // 5 + class number
    const int rows = 25200;     // for some reason net output always looks like this

    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) 
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) 
        { 
            float hand_score = data[5];
            // Continue if the class score is above the threshold.
            if (hand_score > SCORE_THRESHOLD) 
            {
                // Store  confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);

                // Center. yolo notation
                float cx = data[0];
                float cy = data[1];
                // Box dimension. yolo notation
                float w = data[2];
                float h = data[3];

                // Bounding box coordinates. Convert from yolo notation
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += dimensions;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    // Select right bboxes
    vector<Rect> resultingBBoxes;
    for (int i = 0; i < indices.size(); i++)
    {
        convertLetterboxCoords(boxes[indices[i]], img);  // convert bboxes coords
        resultingBBoxes.push_back(boxes[indices[i]]);
    }

    return resultingBBoxes;
}


// reshape the image into one which is (on smaller istances) bigger and square. Image is not stretched but gray bands are added to fill the gaps
Mat letterbox(Mat &img, Size new_shape, Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)     // https://github.com/Hexmagic/ONNX-yolov5
{
    float width = img.cols;
    float height = img.rows;
    float r = min(new_shape.width / width, new_shape.height / height);
    if (!scaleup)
        r = min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
    if (_auto)
    {
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2;
    Mat dst;
    resize(img, dst, Size(new_unpadW, new_unpadH), 0, 0, INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, BORDER_CONSTANT, color); // NAMESPACE NEEDED TO AVOID AMBIGUITY
    return dst;
}


void convertLetterboxCoords(cv::Rect &bbox, cv::Mat &img)
{
    // find translate bbox rect spacial coords. New coords apply on img without padding
    float imgWidth = (float)img.cols, imgHeight = (float)img.rows;
    float scaleRatio = 0.F, padding = 0.F, smallSideSize = 0.F;
    float xPrime = 0.F, yPrime = 0.F;

    float imgAspectRatio = imgWidth / imgHeight;
    float dnnInputAspectRatio = INPUT_IMG_SIZE / INPUT_IMG_SIZE;    // this one is beacuse we hoped to be able to use yolo with image size of 640x320 or something like that

    if (imgAspectRatio >= dnnInputAspectRatio)
    {
        scaleRatio = imgWidth / INPUT_IMG_SIZE;
        smallSideSize = imgHeight / scaleRatio; // big side size is equal to INPUT_WIDTH
        padding = (INPUT_IMG_SIZE - smallSideSize) / 2.F;

        xPrime = (float)bbox.x;
        yPrime = (float)bbox.y - padding;
    }
    else
    {
        scaleRatio = imgHeight / INPUT_IMG_SIZE;
        smallSideSize = imgWidth / scaleRatio; // big side size is equal to INPUT_WIDTH
        padding = (INPUT_IMG_SIZE - smallSideSize) / 2.F;

        xPrime = (float)bbox.x - padding;
        yPrime = (float)bbox.y;
    }
    
    // now to compute correct bbox coords
    bbox.x = (int)round((double)(xPrime * scaleRatio));
    bbox.y = (int)round((double)(yPrime * scaleRatio));
    
    // compute bbox height and width
    bbox.width = (int)round((double)bbox.width * (double)scaleRatio);
    bbox.height = (int)round((double)bbox.height * (double)scaleRatio);

    // check if bbox fit within image
    if (img.cols - bbox.x - bbox.width <= 0)
        bbox.width = img.cols - bbox.x - 1;
    if (img.rows - bbox.y - bbox.height <= 0)
        bbox.height = img.rows - bbox.y - 1;
}


void showBBoxes(Mat img, vector<Rect> bboxes, int imgNum)
{
    Mat displayImg = img.clone();
    for (int i = 0; i < bboxes.size(); i++)
        rectangle(displayImg, bboxes[i], Scalar(0,0,255), 2);
    
    // show image with bounding boxes and rename window according number(helps viewing)
    imshow("bboxDisplay", displayImg);
    setWindowTitle("bboxDisplay", "Localization Results: image n° " + to_string(imgNum));
}

// This method prints both the bbox we obtained and the ground truth
void showBBoxes2(Mat img, vector<Rect> bboxes, vector<Rect> ground_truth,  int imgNum)
{
    Mat displayImg = img.clone();
    for (int i = 0; i < bboxes.size(); i++){
        rectangle(displayImg, bboxes[i], Scalar(0,0,255), 2);
        rectangle(displayImg, ground_truth[i], Scalar(0,255,0), 2);
    }
    // show image with bounding boxes and rename window according number(helps viewing)
    imshow("bboxDisplay", displayImg);
    setWindowTitle("bboxDisplay", "Localization Results: image n° " + to_string(imgNum));
}

double computeIOU(std::vector<cv::Rect> trueBBoxes, std::vector<cv::Rect> bboxes, int rows, int cols)
{
    Mat trueBBoxesRepresentation = Mat::zeros(Size(rows,cols), CV_8UC1);
    Mat bboxesRepresetation = Mat::zeros(Size(rows,cols), CV_8UC1);

    // draw full white bboxes
    for (int i = 0; i < trueBBoxes.size(); i++)
        rectangle(trueBBoxesRepresentation, trueBBoxes[i], (uchar)255U, FILLED);

    for (int i = 0; i < bboxes.size(); i++)
        rectangle(bboxesRepresetation, bboxes[i], (uchar)255U, FILLED);

    Mat intersection;
    bitwise_and(trueBBoxesRepresentation, bboxesRepresetation, intersection);

    Mat union_;
    bitwise_or(trueBBoxesRepresentation, bboxesRepresetation, union_);

    // count number of union pixels
    double unionPxs = 0.;
    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            if (union_.at<uchar>(row,col) == (uchar)255U)
                unionPxs++;

    // count number of intersection pixels
    double intersectionPxs = 0.;
    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            if (intersection.at<uchar>(row,col) == (uchar)255U)
                intersectionPxs++;
    
    return intersectionPxs / unionPxs;
}





// This method returns the score of the bbox passed to it compared to the ground truth
float bbox_IoU(Rect bbox, Rect ground_truth){
    float IoU;

    if (bbox.tl().x > ground_truth.tl().x)
        float xA = bbox.tl().x;
    else
        float xA = ground_truth.tl().x;

    if (bbox.tl().y > ground_truth.tl().y)
        float yA = bbox.tl().y;
    else
        float yA = ground_truth.tl().y;

    return IoU;
}