#include <handLocalization.hpp>
#include <handSegmentation.hpp>

#include <iostream>
#include <string>
#include <filesystem>

using namespace std;

void sortNames(vector<string> & names);   // uses bubble sort since it does not need to sort huge arrays
string getLabelsFilename(string imageFilename, string labelsDirectoryPath);

int main(int argc, char ** argv) 
{
    vector<string> imgPaths;
    for (const auto & entry : filesystem::directory_iterator(argv[1]))
        imgPaths.push_back(entry.path());
    
    sortNames(imgPaths);

    string yoloOutputDir(argv[2]);
    for (int i = 0; i < imgPaths.size(); i++)
    {
        HandData data = loadImgAndBboxes(imgPaths[i], getLabelsFilename(imgPaths[i], string(argv[2])));
        segmentHandsWatershed(data);
    }
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

string getLabelsFilename(string imageFilename, string labelsDirectoryPath)
{
    size_t nameBegin = imageFilename.find_last_of('/') + 1;
    size_t nameLenght = imageFilename.find_last_of('.', nameBegin) + 1;
    
    string name = imageFilename.substr(nameBegin, nameLenght);

    return labelsDirectoryPath + name + ".txt";
}











