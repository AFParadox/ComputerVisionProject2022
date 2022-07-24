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

const string yoloPath = "../src/yolov5/";
const string yoloLabelsDirPath = "../src/yolov5/runs/detect/hand_det/labels/";

const string nnWeights = "../bestM.onnx";

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

const Scalar RED = Scalar(0,0,255);


vector<vector<Rect>> localizeHands(string datasetPath, string weightsPath, vector<string> imgsPath)
{
    // run yolo detect.py
    runYoloDetection(datasetPath, weightsPath);

    vector<vector<Rect>> allDatasetBBoxes;
    for (int i = 0; i < imgsPath.size(); i++)
    {
        Mat img = imread(imgsPath[i]);  // load img just to get width and height
        string labelsFilePath = yoloLabelsDirPath + getLabelsName(imgsPath[i]);
        allDatasetBBoxes.push_back( loadFromYolo(labelsFilePath, (double)img.rows, (double)img.cols) );
    }
    
    return allDatasetBBoxes;
}


void runYoloDetection(string datasetPath, string weightsPath)
{
    // must remove yolo runs directory each time before detection
    if (filesystem::exists(yoloPath + "runs/"))
        filesystem::remove_all(yoloPath + "runs/");

    // write command to detect with yolo
    string detectCommand = "python " + yoloPath + "detect.py --source " + datasetPath + " --weights " + weightsPath + " --conf 0.4 --name hand_det --nosave --save-txt";

    // detect with yolo
    cout << "hand detection starting..." << endl;
    system(detectCommand.c_str());
    cout << endl << endl;
}


string getLabelsName(string imageFilename)
{
    // find where name begins and ends
    size_t nameBegin = imageFilename.find_last_of('/') + 1;
    size_t nameLenght = imageFilename.find_last_of('.', nameBegin) + 1;
    
    // isolate name
    string name = imageFilename.substr(nameBegin, nameLenght);

    return name + ".txt";
}


vector<Rect> loadFromYolo(string yoloLabelsPath, double imgHeight, double imgWidth)
{
    vector<Rect> bboxes;

    if (filesystem::exists(yoloLabelsPath))   // if labels file does not exists means that no hands were detected
    {
        // bboxes from file
        ifstream myFile(yoloLabelsPath);
        int classID;
        double propX, propY, propW, propH; // prop stands for "proportional" since yolo outputs coordinates are all divided by picure size
        int x, y, w, h;

        int j = 0;
        while (myFile >> classID >> propX >> propY >> propW >> propH)
        {
            // convert coordinates from yolo notation to classic one
            w = (int)(imgWidth * propW);
            h = (int)(imgHeight * propH);

            x = (int)(imgWidth * propX) - w / 2;
            y = (int)(imgHeight * propY) - h / 2;

            Rect r(x, y, w, h);
            bboxes.push_back(r);
            j++;
        }
    }

    return bboxes;
}


void saveLabels2PtNotation(vector<string> imgsPath, string saveDir, vector<vector<Rect>> allDatasetBBoxes)
{
    for (int i = 0; i < imgsPath.size(); i++)
    {
        string newLabelsName = saveDir + getLabelsName(imgsPath[i]);
        
        // delete file if it already exists
        if (filesystem::exists(newLabelsName))
            filesystem::remove(newLabelsName);

        // open output file
        ofstream myStream;
        myStream.open(newLabelsName);
        
        for (int j = 0; j < allDatasetBBoxes[i].size(); j++)
            myStream << allDatasetBBoxes[i][j].x << allDatasetBBoxes[i][j].y << allDatasetBBoxes[i][j].width << allDatasetBBoxes[i][j].height << endl;
        
        // close file
        myStream.close();
    }
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








vector<Rect> localizeHands_opencvNN(Mat &img)     // https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python
{
    // Load model.
    Net net;
    net = readNet(nnWeights);

    // not sure if the input need to be agumented with letterbox...
    Mat yoloResizedImg = letterbox(img, Size(640, 640), Scalar(114, 114, 114), false, false, true, 32);

    // Convert to blob.
    Mat blob;
    blobFromImage(yoloResizedImg, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> detections;
    net.forward(detections, net.getUnconnectedOutLayersNames());

    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<float> confidences;
    vector<Rect> boxes; 

    // Resizing factor.
    float x_factor = yoloResizedImg.cols / INPUT_WIDTH;
    float y_factor = yoloResizedImg.rows / INPUT_HEIGHT;

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
    float dnnInputAspectRatio = INPUT_WIDTH / INPUT_HEIGHT;

    if (imgAspectRatio >= dnnInputAspectRatio)
    {
        scaleRatio = imgWidth / INPUT_WIDTH;
        smallSideSize = imgHeight / scaleRatio; // big side size is equal to INPUT_WIDTH
        padding = (INPUT_HEIGHT - smallSideSize) / 2.F;

        xPrime = (float)bbox.x;
        yPrime = (float)bbox.y - padding;
    }
    else
    {
        scaleRatio = imgHeight / INPUT_HEIGHT;
        smallSideSize = imgWidth / scaleRatio; // big side size is equal to INPUT_WIDTH
        padding = (INPUT_WIDTH - smallSideSize) / 2.F;

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