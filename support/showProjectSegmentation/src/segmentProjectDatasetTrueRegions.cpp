#include <segmentProjectDatasetTrueRegions.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void segmentProjectDatasetTrueRegions::printCommands()
{
    cout << "Press 'q' or ESC key to quit (or exit current presentation)" << endl;
    cout << "Press 'd' to move to next picture "  << endl;
    cout << "Press 'a' to move to previous picture "  << endl;
}

segmentProjectDatasetTrueRegions::segmentProjectDatasetTrueRegions()
{
    imgsLoaded = false;
    handBoxesLoaded = false;
    trueMasksLoaded = false;
}

segmentProjectDatasetTrueRegions::~segmentProjectDatasetTrueRegions()
{
}

int segmentProjectDatasetTrueRegions::loadImgsWithGlob(char *pattern)
{
    vector<string> filenames;
    glob(pattern, filenames, true);

    // check if the path is correct
    if (filenames.size() < 1)
        throw invalid_argument("There are no images inside the specified directory with the specified pattern");

    // empty vector if already loaded (also empty the other vectors)
    if (imgsLoaded)
    {
        imgs.clear(); imgs.shrink_to_fit(); imgsLoaded = false;

        if (handBoxesLoaded)
        { handBoxes.clear(); handBoxes.shrink_to_fit(); handBoxesLoaded = false; }

        if(trueMasksLoaded)
        { trueMasks.clear(); trueMasks.shrink_to_fit(); trueMasksLoaded = false; }
    }

    // load imgs into vector
    imgs.resize(filenames.size());
    for (int i = 0; i < filenames.size(); i++)
        imgs[i] = imread(filenames[i]);
    
    imgsLoaded = true;

    return imgs.size();
}

int segmentProjectDatasetTrueRegions::loadHandBoxesCoordsWithGlob(char *pattern)
{
    vector<string> filenames;
    glob(pattern, filenames, true);

    // check if the path is correct
    if (filenames.size() < 1)
        throw invalid_argument("There are no images inside the specified directory with the specified pattern");
    // check if handBoxes and imgs correspons
    if (!imgsLoaded)
        throw invalid_argument("Must load image first");
    if (imgs.size() != filenames.size())
        throw invalid_argument("n° of images and n° of handBoxes do not correspond");
    
    // empty vector if already loaded
    if (handBoxesLoaded)
    { handBoxes.clear(); handBoxes.shrink_to_fit(); }

    // load coordinates
    int x, y, w, h;
    handBoxes.resize(filenames.size());
    
    for (int i = 0; i < filenames.size(); i++)
    {
        ifstream myFile(filenames[i].c_str());
        int j = 0;
        while (myFile >> x >> y >> w >> h)
        {
            Rect r(x, y, w, h);
            handBoxes[i].push_back(r);
            j++;
        }
    }

    int handCount = 0;
    for (int i = 0; i < handBoxes.size(); i++)
        handCount += handBoxes[i].size();
    
    handBoxesLoaded = true;

    /*/ DEBUG print hand boxes coordinates
    for (int i = 0; i < handBoxes.size(); i++)
    {
        cout << "img " << i << ":" << endl;
        for (int j = 0; j < handBoxes[i].size(); j++)
            cout << handBoxes[i][j].x << ", " << handBoxes[i][j].y << ";    " << handBoxes[i][j].width << ", " << handBoxes[i][j].height << endl;
        cout << endl;
    } // */

    return handCount;
}

int segmentProjectDatasetTrueRegions::loadTrueSegmentationMask(char *pattern)
{
    vector<string> filenames;
    glob(pattern, filenames, true);

    // check if the path is correct
    if (filenames.size() < 1)
        throw invalid_argument("There are no images inside the specified directory with the specified pattern");
    // check if handBoxes and imgs has been loaded(No point in these without box images)
    if (!handBoxesLoaded || !imgsLoaded)
        throw invalid_argument("you must load first images and box coordinates");
    // check if n° of masks and n° of imgs correspons
    if (imgs.size() != filenames.size())
        throw invalid_argument("n° of images and n° of masks do not correspond");

    if (trueMasksLoaded)
    { trueMasks.clear(); trueMasks.shrink_to_fit(); trueMasksLoaded = false; }
    
    trueMasks.resize(imgs.size());

    for (int i = 0; i < filenames.size(); i++)
        trueMasks[i] = imread(filenames[i], IMREAD_GRAYSCALE);
    
    trueMasksLoaded = true;

    return trueMasks.size();
}

void segmentProjectDatasetTrueRegions::showImgsWithLocalizedHands()
{
    if (!imgsLoaded || !handBoxesLoaded)
        throw invalid_argument("must load both images and hand coordinates");

    namedWindow("img");
    Mat img;
    char nxt = 't';
    int i = 0;
    
    printCommands();

    do
    {
        if ((nxt == 'd') && (i < imgs.size()-1))
            i++;
        else if ((nxt == 'a') && (i > 0))
            i--;
        else if ((nxt == 'q') || (nxt == 27))
            break;

        img = imgs[i].clone();
        for (int j = 0; j < handBoxes[i].size(); j++)
            rectangle(img, handBoxes[i][j], Scalar(0,0,255), 2);
        
        imshow("img", img);
        setWindowTitle("img", "image " + to_string(i));

    } while (nxt = (char)waitKey(0));
    
}

void segmentProjectDatasetTrueRegions::showImgsWithTrueMasksSeg()
{
    if (!imgsLoaded || !handBoxesLoaded || !trueMasksLoaded)
        throw invalid_argument("must load both images and hand coordinates");
    
    namedWindow("img");
    Mat img;
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

        // draw objective segmentation
        img = segmentImgWithTrueMask(i);

        // draw box localization
        for (int j = 0; j < handBoxes[i].size(); j++)
            rectangle(img, handBoxes[i][j], Scalar(0,0,255), 2);

        imshow("img", img);
        setWindowTitle("img", "image " + to_string(i));

    } while (nxt = (char)waitKey(0));
}

Mat segmentProjectDatasetTrueRegions::segmentImgWithTrueMask(int imgID)
{
    Vec3b color((uchar)theRNG().uniform(0,127), (uchar)theRNG().uniform(0,127), (uchar)theRNG().uniform(0,127));

    Mat segmented = imgs[imgID].clone();
    Mat cropImg, cropMask;
    for (int i = 0; i < handBoxes[imgID].size(); i++)
    {
        cropImg = segmented(handBoxes[imgID][i]);
        cropMask = trueMasks[imgID](handBoxes[imgID][i]);

        for (int row = 0; row < cropMask.rows; row++)
            for (int col = 0; col < cropMask.cols; col++)
                if (cropMask.at<uchar>(row,col) == 255)
                    cropImg.at<Vec3b>(row,col) = cropImg.at<Vec3b>(row,col)/2 + color;
    }

    return segmented;
}

