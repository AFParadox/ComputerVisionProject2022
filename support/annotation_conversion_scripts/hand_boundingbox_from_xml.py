import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2
from IPython.display import Image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

annotations_path = '/home/leonardo/Documents/Unipd_Magistrale/1.2_computer_vision/Project/hand_over_face/annotations/'
images_path = '/home/leonardo/Documents/Unipd_Magistrale/1.2_computer_vision/Project/hand_over_face/images_original_size/'
new_annotations_path = '/home/leonardo/Documents/Unipd_Magistrale/1.2_computer_vision/Project/hand_over_face/yolov5_annotations/'

for filename in os.listdir(annotations_path):

    print('\n')
    print(filename)
    annotation_fullname = annotations_path + filename
    image_fullname = images_path + filename[:-4] + '.jpg'

    #Image shape isn't in all xml files, so we check jpg files
    img = cv2.imread(image_fullname)
    width = img.shape[1]
    height = img.shape[0]
    print('Img shape:', width, height)

    #Get root element of xml file    
    tree = ET.parse(annotation_fullname)
    root = tree.getroot()

    #Get size of the image                      // Not all xml files carry image size
    #imagesize = root.find('imagesize')         // so this part is no longer used
    #height = float(imagesize[0].text)
    #width = float(imagesize[1].text)
    #print('Image size:', width, 'x', height)


    obj = root.findall('object')
    for element in obj:

        polyg = element.find('polygon')

        #Create objects for the for points of the bounding box
        left = Point(
            float(polyg[1][0].text),
            float(polyg[1][1].text))
        right = Point(
            float(polyg[1][0].text),
            float(polyg[1][1].text))
        top = Point(
            float(polyg[1][0].text),
            float(polyg[1][1].text))
        bottom = Point(
            float(polyg[1][0].text),
            float(polyg[1][1].text))

        #Search for the correct value of the points
        for child in polyg:
            if child.tag == 'pt':
                x = float(child.find('x').text)
                y = float(child.find('y') .text)
                if left.x > x:
                    left.x = x
                    left.y = y
                if right.x < x:
                    right.x = x
                    right.y = y
                if top.y < y:
                    top.x = x
                    top.y = y
                if bottom.y > y:
                    bottom.x = x
                    bottom.y = y
        #print('Left: ', left.x, ' ',left.y)
        #print('Right: ', right.x, ' ',right.y)
        #print('Top: ', top.x, ' ',top.y)
        #print('Bottom: ', bottom.x, ' ',bottom.y)

        #Compute bbox size and center coordinates
        bbox_width = right.x - left.x
        bbox_height = top.y - bottom.y
        center_x = left.x + bbox_width/2
        center_y = bottom.y + bbox_height/2

        print('Pre-normalized values: {:.6f} {:.6f} {:.6f} {:.6f}'.format(center_x, center_y, bbox_width, bbox_height))

        #Normalization of bbox size and center coordinates
        bbox_width /=  width
        bbox_height /= height
        center_x /= width
        center_y /= height

        #print('Bbox center: {:.6f} {:.6f}'.format(center_x, center_y))
        print('0 {:.6f} {:.6f} {:.6f} {:.6f}'.format(center_x, center_y, bbox_width, bbox_height))

        #Write results in .txt file
        new_annotation_fullname = new_annotations_path + filename[:-4] + '.txt'
        with open(new_annotation_fullname, 'a') as f:
            f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(center_x, center_y, bbox_width, bbox_height))

        #new_filename = filename, '.txt'
        #txt = open(new_filename, 'w+')