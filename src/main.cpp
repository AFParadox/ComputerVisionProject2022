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
const string trueMasksPath = "../docs/evaluationDataset/mask/";
const string datasetPath = "../docs/evaluationDataset/rgb/";
const string modelPath = "../best_img640_batch20_epochs60_M.onnx";


void sortNames(vector<string> & names);   // simple sorting algorithm
void resultsSlideshow(vector<Mat> imgs, vector<vector<Rect>> allDatasetBBoxes, vector<Mat> masks);   // display results nicely

double scoreLocalization(string trueCoordFilename, vector<Rect> bboxes, int rows, int cols);
double scoreSegmentation(string trueMaskFilename, Mat mask);


int main(int argc, char ** argv) 
{
    // load each image path into vector
    vector<string> imgsPath;
    for (const auto & entry : filesystem::directory_iterator(datasetPath))
        imgsPath.push_back(entry.path());
    
    // sort them
    sortNames(imgsPath);

    // load images into memory
    vector<Mat> imgs;
    for (int i = 0; i < imgsPath.size(); i++)
        imgs.push_back(imread(imgsPath[i]));

    // load yolov5 model
    dnn::Net yolov5Model = loadModel(modelPath);

    system("clear");    // hoping you are using linux :)
    // print commands
    cout << "Press 'q' or ESC key to quit (or exit current presentation)" << endl;
    cout << "Press 'd' to move to next picture "  << endl;
    cout << "Press 'a' to move to previous picture "  << endl;

    char nxt = 't';
    int i = 0;

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

        //saveHandIstances(to_string(i), imgs[i], bboxes, "../../../Documents/hands_cropped/");

        showBBoxes(imgs[i], bboxes, i);
        showSegmentedHands(imgs[i], mask, i);

        // compute localization score
        string trueBBoxesPath = annotationsPath + imgsPath[i].substr(imgsPath[i].find_last_of('/')+1, 2) + string(".txt");
        double currentLocScore = scoreLocalization(trueBBoxesPath, bboxes, imgs[i].rows, imgs[i].cols);

        // compute pixel accuracy score
        string currentMaskPath = trueMasksPath + imgsPath[i].substr(imgsPath[i].find_last_of('/')+1, 2) + string(".png");
        double currentSegmScore = scoreSegmentation(currentMaskPath, mask);

        // print score
        cout << "Image " << to_string(i) << " has bounding boxes IoU = " << to_string(currentLocScore) << " and segmentation pixel accuracy of " << to_string(currentSegmScore) << endl;

    } while (nxt = (char)waitKey(0));

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



double scoreLocalization(string trueCoordFilename, vector<Rect> bboxes, int rows, int cols)
{
    // load file
    // load annotations into memory
    vector<Rect> trueLabels;

    ifstream labelFile;
    labelFile.open(trueCoordFilename);
    if (!labelFile) // check if file has been opened succesfully
    {
        cout << "Unable to open " << trueCoordFilename << endl;
        exit(1);
    }
    string bbox_str;

    int x, y, w, h;
    while (labelFile >> x >> y >> w >> h)
        trueLabels.push_back(Rect(x, y, w, h));

    return computeIOU(trueLabels, bboxes, rows, cols);
}


double scoreSegmentation(string trueMaskFilename, Mat mask)
{
    Mat trueMask = imread(trueMaskFilename, IMREAD_GRAYSCALE);

    return computePixelAccuracyScore(mask, trueMask);
}


