#include <handLocalization.hpp>
#include <handSegmentation.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;

const string annotationsPath = "../docs/evaluationDataset/det/";
const string datasetPath = "../docs/evaluationDataset/rgb/";
const string modelPath = "../best_img512_batch10_epochs120_L.onnx";


void sortNames(vector<string> & names);   // simple sorting algorithm
void resultsSlideshow(vector<Mat> imgs, vector<vector<Rect>> allDatasetBBoxes, vector<Mat> masks);   // display results nicely

void saveAllHandIstancesCropped(vector<string> imgPaths, string yoloOutputDir, string saveDirLocation);     // save hand instances in sigle files(useful when trying things for segmentation)


int main(int argc, char ** argv) 
{
    // load each image path into vector
    vector<string> imgsPath;
    for (const auto & entry : filesystem::directory_iterator(datasetPath))
        imgsPath.push_back(entry.path());
    
    // load each annotation path into vector
    vector<string> labelsPath;
    for (const auto & entry: filesystem::directory_iterator(annotationsPath))
        labelsPath.push_back(entry.path());
    
    // sort them
    sortNames(imgsPath);
    sortNames(labelsPath);

    // load images into memory
    vector<Mat> imgs;
    for (int i = 0; i < imgsPath.size(); i++)
        imgs.push_back(imread(imgsPath[i]));
    
    // load annotations into memory
    vector<Rect> labels;
    string delimiter = " ";
    int center_x;
    int center_y;
    int width;
    int height;
    for (int i = 0; i < labelsPath.size(); i++){
        ifstream labelFile;
        labelFile.open(labelsPath[i]);
        if (!labelFile){
            cout << "Unable to open " << labelsPath[i] << endl;
            exit(1);
        }
        string bbox;
        while(getline(labelFile, bbox)){
            center_x = stoi(bbox.substr(0, bbox.find(delimiter)));
            center_y = stoi(bbox.substr(1, bbox.find(delimiter)));
            width = stoi(bbox.substr(2, bbox.find(delimiter)));
            height = stoi(bbox.substr(3, bbox.find(delimiter)));
            cout << center_x << ' ' << center_y << ' ' << width << ' ' << height << endl; //PORCODDIO NON VA
        }
        cout << endl;
    }

    // load yolov5 model
    dnn::Net yolov5Model = loadModel(modelPath);

    system("clear");    // hoping you are using linux :)
    // print commands
    cout << "Press 'q' or ESC key to quit (or exit current presentation)" << endl;
    cout << "Press 'd' to move to next picture "  << endl;
    cout << "Press 'a' to move to previous picture "  << endl;

    char nxt = 't';
    int i = 0;
    Vec3b handColor = Vec3b(0,0,200);   // kind of red

    do
    {
        if ((nxt == 'd') && (i < imgs.size()-1))    // go to next image with 'd' key
            i++;
        else if ((nxt == 'a') && (i > 0))           // go to previous image with 'a' key
            i--;
        else if ((nxt == 'q') || (nxt == 27))       // exit by either pressing ESCAPE key or 'q'
            break;

        vector<Rect> bboxes = localizeHands(imgs[i], yolov5Model);
        Mat mask = segmentHandsWatershed(imgs[i], bboxes);

        showBBoxes(imgs[i], bboxes, i);
        showSegmentedHands(imgs[i], mask, i, handColor);

    } while (nxt = (char)waitKey(0));

    exit(EXIT_SUCCESS); // not necessary but why not

    return 0;
}



void sortNames(vector<string>& names)
{
    bool iteration = true;
    for (int i = 0; i < names.size(); i++)
    {
        for (int j = names.size()-1; j > i; j--)
        {
            if (names[j].compare(names[j-1]) < 0)
            {
                iteration = true;
                string temp = names[j];
                names[j] = names[j-1];
                names[j-1] = temp;
            }
        }
        if (!iteration) break;
    }
}



