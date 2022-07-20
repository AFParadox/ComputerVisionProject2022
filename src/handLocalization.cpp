#include <handLocalization.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

const string yoloPath = "../src/yolov5/";
const string yoloLabelsDirPath = "../src/yolov5/runs/detect/hand_det/labels/";


vector<vector<Rect2i>> localizeHands(string datasetPath, string weightsPath, vector<string> imgsPath)
{
    // run yolo detect.py
    runYoloDetection(datasetPath, weightsPath);

    vector<vector<Rect2i>> allDatasetBBoxes;
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


vector<Rect2i> loadFromYolo(string yoloLabelsPath, double imgHeight, double imgWidth)
{
    vector<Rect2i> bboxes;

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

            Rect2i r(x, y, w, h);
            bboxes.push_back(r);
            j++;
        }
    }

    return bboxes;
}


void saveLabels2PtNotation(vector<string> imgsPath, string saveDir, vector<vector<Rect2i>> allDatasetBBoxes)
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

void showBBoxes(Mat img, vector<Rect2i> bboxes, int imgNum)
{
    Mat displayImg = img.clone();
    for (int i = 0; i < bboxes.size(); i++)
        rectangle(displayImg, bboxes[i], Scalar(0,0,255), 2);
    
    // show image with bounding boxes and rename window according number(helps viewing)
    imshow("bboxDisplay", displayImg);
    setWindowTitle("bboxDisplay", "Localization Results: image n° " + to_string(imgNum));
}


