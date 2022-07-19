from scipy.io import loadmat
import numpy as np
import glob
import os
import cv2

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def toString(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'


# Here we load the .mat file that we want to convert
file = loadmat('useful_files/polygons.mat')

polygon = file['polygons'][0]

# TO UPDATE
photos_directory = '/home/leonardo/Documents/Unipd_Magistrale/1.2_computer_vision/Project/egohands_data/usable_FORSURE/PUZZLE_OFFICE_T_S/'
photo_index = 0

# TO UPDATE
annotation_directory = '/home/leonardo/Documents/Unipd_Magistrale/1.2_computer_vision/Project/egohands_data/yolov5_annotations/PUZZLE_OFFICE_T_S/'

counter = 39000
for infile in sorted(glob.glob(photos_directory + '*.jpg')):

    new_photo_name = photos_directory + str(counter) + '.jpg'

    # Get image shape
    img = cv2.imread(infile)
    width = img.shape[1]
    height = img.shape[0]

    # Create new annotation file name (for every folder we initialize a new counter so that all files have different names)
    img_name = infile[-14:-4]
    new_annotation_filename = str(counter) + '.txt'
    new_annotation_path = annotation_directory + new_annotation_filename
    print()
    print(img_name)
    print('img shape:', width, height)

    # Create new annotations file
    f = open(new_annotation_path, 'x')

    for hand in polygon[photo_index]:

        print(hand.size)
        # Check if the array is empty (this means that current hand is not in the photo)
        if hand.size == 0:
            continue

        # Create four points
        left = Point(hand[0][0], hand[0][1])
        right = Point(hand[0][0], hand[0][1])
        top = Point(hand[0][0], hand[0][1])
        bottom = Point(hand[0][0], hand[0][1])

        for point in hand:
            if left.x > point[0]:
                left.x = point[0]
                left.y = point[1]
            if right.x < point[0]:
                right.x = point[0]
                right.y = point [1]
            if top.y < point[1]:
                top.x = point[0]
                top.y = point[1]
            if bottom.y > point[1]:
                bottom.x = point[0]
                bottom.y = point[1]

        # Computing bbox center, width and height, plus normalizing
        bbox_width = right.x - left.x
        bbox_height = top.y - bottom.y
        center_x = left.x + bbox_width/2
        center_y = bottom.y + bbox_height/2

        bbox_width /= width
        bbox_height /= height
        center_x /= width
        center_y /= height

        print('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(center_x, center_y, bbox_width, bbox_height))
        f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(center_x, center_y, bbox_width, bbox_height))

    os.rename(infile, new_photo_name)
    photo_index += 1
    counter += 1
