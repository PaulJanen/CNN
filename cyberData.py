"""
Slightly modified version of https://github.com/hsjeong5/MNIST-for-Numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

def load():
    imgAllLetters = Image.open("PS\AllLetters.png").convert('L')
    imgWidth = 28
    width, height = imgAllLetters.size
    cuts = int(width/imgWidth)
    images = []
    for i in range(cuts):
        x = i*imgWidth
        y = 0
        boxWidth = imgWidth
        boxHeight = height
        img = imgAllLetters.crop((x, y, x+boxWidth, boxHeight))
        images.append(img)

    XTrain = (np.array(images[0])).flatten()
    for i in range(1,cuts):
        img = (np.array(images[i])).flatten()
        XTrain = np.row_stack((XTrain,img)) 
    YTrain = np.array(([0,1,2,3,4]))
    cyber = {}
    cyber["training_images"] = XTrain
    cyber["training_labels"] = YTrain
    cyber["test_images"] = XTrain
    cyber["test_labels"] = YTrain
    cyber["prediction_image"] = np.array(imgAllLetters).reshape((1,height,width,1))
    return cyber["training_images"], cyber["training_labels"], cyber["test_images"], \
            cyber["test_labels"], cyber["prediction_image"]