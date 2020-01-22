import numpy as np
import pandas as pd
from imutils import paths
import cv2



def move_data_to_datasets():
    imgPaths= list(paths.list_files("../do_An_3/dataset/"))
    for imgPath in imgPaths:
        filename = imgPath.split("/")[-1]
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        cv2.imwrite("../finall/datasets/" + filename, img)

move_data_to_datasets()
    