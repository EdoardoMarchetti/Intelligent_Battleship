import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter

from utilities import *
from BattleField import *

#######################################################
path = "Photos\\"
"""...............Configuration Param................"""
#Immagini lette bene scritte a mano su carta
path_image_pennarello_4 = path + "pennarello_4.jpeg"
path_image_pennarello_5 = path + "pennarello_5.jpeg"
path_image_pennarello_1 = path + "pennarello_1.jpeg"
path_image_pennarello_3 = path + "pennarello_3.jpeg"
path_image_4z= "Photos\\4z.jpg" 
path_image_6z= "Photos\\6z.jpg" 
path_image_arancio = path +"arancio.jpeg"

#Immagini lette bene scritte a mano su whiteboard
path_image_wb_2 = path + "wb_2.jpg" 
path_image_black_wb3 = path + "black_wb3.png"
path_image_black_wb2 = path + "black_wb2.png"
path_image_black_wb = path + "black_wb.png"
path_image_blu_wb2 = path + "blu_wb2.png"


#Immagini lette bene scritte da computer su whiteboard
path_image_wb_d1 = path + "wb_d1.jpg"
path_image_wb_d2 = path + "wb_d2.jpg"
path_image_wb_w = path + "wb_w.jpg"


#Immagini non lette bene
path_image_2z= "Photos\\2z.jpg" #errore di lettura dei 4
path_image_5z= "Photos\\5z.jpg" #errore di lettura dei 4
path_image_black_wb4 = path + "black_wb4.png" #errore di lettura di un 5 e di un 4
path_image_wb_1 = path + "wb_1.jpg" #errore di lettura di un 4
path_image_wb_4 = path + "wb_4.jpg" #errore di lettura su un 2 e un 5



heightImg = 450
widthImg = 450
model = initialiazePredictionModel() #Load the cnn model
#modelTest(model) utility function to verify if the model is loaded
#######################################################


# 1 Prepare the image
img = cv.imread(path_image_wb_4)

img = cv.resize(img, (widthImg,heightImg)) #resize to have a square image
imgBlank = np.zeros((heightImg, widthImg,3) , dtype = 'uint8' )
imgTreshold = preProcess(img)

cv.imshow('Original', img)
# cv.imshow('treshold', imgTreshold)

# 2 find all the contours
imgContours = img.copy()
imgBigContours = img.copy()
contrours , hierarchy = cv.findContours(imgTreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(imgContours, contrours, -1, (0,255,0), 3)
# cv.imshow('Contours', imgContours)

# 3 Find the biggest contour and use it as sudoku
biggest, maxArea = biggestContour(contrours) # Find the biggest square or rectangle contour
#print(biggest)
if biggest.size != 0 :
    biggest = reorder(biggest)
    #print("\nAfetr reorder = \n",biggest)
    biggest = addMargin(biggest, widthImg)
    #print("\nAfetr resize = \n",biggest)
    cv.drawContours(imgBigContours, biggest, -1, (0,0,255), 10) # draw the biggest contour
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg,0], [0,heightImg], [widthImg, heightImg]])
    matrix = cv.getPerspectiveTransform(pts1, pts2) #Permette di ottenere un immagine perpendicolare
    imgWrapColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))
    imgForSplit = cv.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    

imgWrapColoredCopy = imgWrapColored.copy()

#cv.imshow('WarpPerspective', imgWrapColored)
cv.imshow('Bigcontour', imgBigContours)


#3.2 Line Detection with HoughLines
imgTreshold = preProcess(imgWrapColored)
dialeted = cv.dilate(imgTreshold, (3,3), iterations = 1)
lines = cv.HoughLines(imgTreshold, 1, np.pi / 90, 280)
points = []

for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        amp = widthImg
        x1 = int(x0 + amp * (-b))
        y1 = int(y0 + amp * (a))
        x2 = int(x0 - amp * (-b))
        y2 = int(y0 - amp * (a))
        cv.line(imgWrapColored, (x1,y1), (x2,y2), (0,0,255), 2)
        points.append([[x1,y1], [x2,y2]])
        

#cv.imshow('Dialate', dialeted)
cv.imshow('Lines with HL', imgWrapColored )
        
        
# 3.3 Get the right lines
[horizontal_points, vertical_points] = getRightLines(points)

#print("Num horizontal =", len(horizontal_points))
#print("Horizontal = ", horizontal_points)

#print("Num Vertical = ", len(vertical_points))
# print("Vertical = ", vertical_points)





#cv.imshow('WrapCopy', imgWrapColoredCopy)

#3.4 Draw the line over the image
ref = imgWrapColoredCopy
for p in horizontal_points:
    x1 = p[0][0]
    x2 = p[1][0]
    y1 = p[0][1]
    y2 = p[1][1]
    cv.line(ref, (x1,y1), (x2,y2), (0,0,255), 2)

for p in vertical_points:
    x1 = p[0][0]
    x2 = p[1][0]
    y1 = p[0][1]
    y2 = p[1][1]
    cv.line(ref, (x1,y1), (x2,y2), (0,0,255), 2)


#cv.imshow('WrapCopy with calculated lines', ref)
# cv.imshow('ImageForSplit', imgForSplit)



### 4 Split the image
boxes = splitBoxes(imgForSplit, horizontal_points, vertical_points)


printBoxes(boxes)
# cv.imshow('Box', immagine)

### 4.1 Get Predictions
[numbers, probs] = getPredictions(boxes, model)


# 4.2 Create the grid
dim = int(np.sqrt(len(numbers)))
grid = np.reshape(numbers, (dim, dim))
probs = np.reshape(probs, (dim,dim))
print("\n\n..................GRIGLIA OTTENUTA..................")
print("Elementi : \n", grid)
print("Probabilità : \n", probs)
grid = correctGrid(grid, probs)
print("\n\n..................GRIGLIA CORRETTA..................")
print(grid)


# 5 Launch the agent
bf = BattleField(500, grid)
bf.show()




















cv.waitKey(0)
cv.destroyAllWindows()










