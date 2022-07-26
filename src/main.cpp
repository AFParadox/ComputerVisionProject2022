#include <handLocalization.hpp>
#include <handSegmentation.hpp>

#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;

const string datasetPath = "../docs/evaluationDataset/rgb/";
const string weightsPath = "../yoloWeights.pt";


void sortNames(vector<string> & names);   // simple sorting algorithm
void resultsSlideshow(vector<Mat> imgs, vector<vector<Rect>> allDatasetBBoxes, vector<Mat> masks);   // display results nicely

void saveAllHandIstancesCropped(vector<string> imgPaths, string yoloOutputDir, string saveDirLocation);     // save hand instances in sigle files(useful when trying things for segmentation)


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

        vector<Rect> bboxes = localizeHands_opencvNN(imgs[i]);
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


void saveAllHandIstancesCropped(vector<string> imgPaths, string yoloOutputDir, string saveDirLocation)
{
    for (int i = 0; i < imgPaths.size(); i++)
    {
        // get image name
        size_t nameBegin = imgPaths[i].find_last_of('/') + 1;
        size_t nameLenght = imgPaths[i].find_last_of('.', nameBegin) + 1;
        string name = imgPaths[i].substr(nameBegin, nameLenght);

        // load image
        //HandData data = loadImgAndBboxes(imgPaths[i], getLabelsFilename(imgPaths[i], yoloOutputDir));

        // save hand istances
        //saveHandIstances(name, data, saveDirLocation);
    }
}



