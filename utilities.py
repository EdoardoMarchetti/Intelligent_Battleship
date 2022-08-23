import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


##Initialize model
def initialiazePredictionModel(): #Permette di caricare un modello per il riconoscimento dell'immagine
    model = tf.keras.models.load_model('cnn_model_3')
    return model

def modelTest(model): #Permette di effettuare una prova per verificare se il modello è caricato correttamente
    mnist = tf.keras.datasets.mnist
 
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    #Normalize
    train_images = tf.keras.utils.normalize(train_images, axis = 1)
    test_images = tf.keras.utils.normalize(test_images, axis = 1)

    predictions = model.predict([test_images])

    print(np.argmax(predictions[10]))
    plt.imshow(test_images[10])
    plt.show()


### 1 - Preprocessing image
def preProcess(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                  # to gray scale
    blur = cv.GaussianBlur(gray, (5,5), 1)                      # add blur
    threshold = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)    #apply adaptive treshold
    return threshold

def preProcessBoxes(img):
                     
    blur = cv.GaussianBlur(img, (7,7), 3)                      # add blur
    erode = cv.erode(blur, (5,5), iterations= 1)
    dialeted = cv.dilate(erode, (5,5), iterations = 3)
    (thresh, im_bw) = cv.threshold(img, 128, 255, cv.THRESH_TRUNC)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # to gray scale
    threshold = cv.adaptiveThreshold(gray, 255, 1, 1, 11, 2)    #apply adaptive treshold
    return threshold

### 3 Find Biggest Contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0

    for i in contours:
        area = cv.contourArea(i)
        if area > 100 :
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True) # approssimazione della curva iniziale i 
            if area > max_area and len(approx) == 4: # len == 4 per ricavare solo quadrati o rettangoli
                biggest = approx
                max_area = area
    
    return biggest, max_area

### 3 Reorder points for warp perspective
def reorder(myPoints):
    #print('In reorder')
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), dtype = np.int32)
    add = myPoints.sum(1)
    #print('Add = ',add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis = 1)
    #print('Diff = ',diff)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

### 3 Transform biggest in a square
def addMargin(myPoints, imgDimension):
    
    p1 = myPoints[0]
    p2 = myPoints[1]
    p3 = myPoints[2]
    p4 = myPoints[3]

    margin=10

    p1 = [p1[0][0]-margin, p1[0][1]-margin]
    p2 = [p2[0][0]+margin, p2[0][1]-margin]
    p3 = [p3[0][0]-margin, p3[0][1]+margin]
    p4 = [p4[0][0]+margin, p4[0][1]+margin]

    print("P1= ", p1)
    print("P1[0]= ",p1[0])
    print("P1[1]= ",p1[1])

    if p1[0] < 0:
        p1[0] = 0
    
    if p1[1]< 0:
        p1[1] = 0

    if p2[0] > imgDimension:
        p2[0] = imgDimension
    
    if p2[1] < 0:
        p2[1] = 0

    if p3[0] < 0:
        p3[0] = 0
    
    if p3[1] > imgDimension:
        p3[1] = imgDimension

    if p4[0] > imgDimension:
        p4[0] = imgDimension
    
    if p4[1] > imgDimension:
        p4[1] = imgDimension

    newPoints = np.zeros((4,1,2), dtype = np.int32)
    newPoints[0] = p1
    newPoints[1] = p2
    newPoints[2] = p3
    newPoints[3] = p4

    return newPoints


### 3.2 Draw lines
def drawLines(img, lines):
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

### 3.3 Get the right Lines
def array_diff(original, to_remove):
        i = 0 

        while i < len(to_remove):
            j = 0
            removed = False
            while j < len(original) and not removed:
                if(original[j] == to_remove[i]):
                    original.remove(original[j])
                    removed = True
                if not removed:
                    j = j+1
            i = i+1
        return original

def getRightLines(points):
    vertical_points = []
    horizontal_points = []

    #Classify by slope
    for coord in points:
        x1 = coord[0][0]
        x2 = coord[1][0]
        y1 = coord[0][1]
        y2 = coord[1][1]

        if x2-x1 != 0:
            slope = (y2-y1) / (x2-x1)
            if slope < 0.1 and slope > -0.1:
                horizontal_points.append([[x1,y1], [x2,y2]])
            elif slope > 10 or slope < -10:
                vertical_points.append([[x1,y1], [x2,y2]])
        else:
            vertical_points.append([[x1,y1], [x2,y2]])

    vertical_points = sorted(vertical_points, key = lambda x : x[0][0])
    horizontal_points = sorted(horizontal_points, key = lambda x : x[0][1])

    #Classify by distance
    dist = 24

    #Vertical lines
    i = 0
    vertical_points_remove = []
    while i < len(vertical_points)-1:
        
        #Get points
        [[fx1, fy1], [fx2, fy2]] = vertical_points[i]

        j = i+1
        remove = False
        while vertical_points_remove.count([[fx1, fy1], [fx2, fy2]]) == 0 and j < len(vertical_points) and not remove:
            
            [[sx1, sy1], [sx2, sy2]]= vertical_points[j]
           

            if abs( (fx1+fx2) / 2 - (sx1+sx2) / 2 ) < dist:
                if fx1-fx2 == 0 :
                    vertical_points_remove.append([[sx1, sy1], [sx2, sy2]])           
       
                elif sx1-sx2 == 0:
                    vertical_points_remove.append([[fx1, fy1], [fx2, fy2]])
                    remove = True
                   
                else:
                    slope_f = (fy2-fy1) / (fx1-fx2)
                    slope_s = (sy2-sy1) / (sx2-sx1)
                    if slope_f > slope_s: 
                        vertical_points_remove.append([[sx1, sy1], [sx2, sy2]]) 
                        
                    elif slope_s > slope_f:
                        
                        vertical_points_remove.append([[fx1, fy1], [fx2, fy2]])
                        remove = True
                        
            j = j+1
        i = i+1

    vertical_points = array_diff(vertical_points, vertical_points_remove)


    #Horizontal lines
    i = 0
    horizontal_points_remove = []
    while i < len(horizontal_points)-1:
        
        #Get points
        [[fx1, fy1], [fx2, fy2]] = horizontal_points[i]

        j = i+1
        remove = False
        while horizontal_points_remove.count([[fx1, fy1], [fx2, fy2]]) == 0 and j < len(horizontal_points) and not remove:
            
            [[sx1, sy1], [sx2, sy2]]= horizontal_points[j]
           

            if abs( (fy1+fy2) / 2 - (sy1+sy2) / 2 ) < dist:
                if fy1-fy2 == 0 :
                    horizontal_points_remove.append([[sx1, sy1], [sx2, sy2]])           
       
                elif sy1-sy2 == 0:
                    horizontal_points_remove.append([[fx1, fy1], [fx2, fy2]])
                    remove = True
                   
                else:
                    slope_f = (fy2-fy1) / (fx1-fx2)
                    slope_s = (sy2-sy1) / (sx2-sx1)
                    if slope_f > slope_s: 
                        horizontal_points_remove.append([[sx1, sy1], [sx2, sy2]]) 
                        
                    elif slope_s > slope_f:
                        
                        horizontal_points_remove.append([[fx1, fy1], [fx2, fy2]])
                        remove = True
                        
            j = j+1
        i = i+1

    horizontal_points = array_diff(horizontal_points, horizontal_points_remove)



    return [horizontal_points, vertical_points]


    


### 4 to split the image
def splitBoxes(img, hp, vp):
    boxes = []
    for i in range(1, len(hp)):
        h1 = hp[i-1]
        h2 = hp[i]
        for j in range(1, len(vp)):
            v1 = vp[j-1]
            v2 = vp[j]
            
            p1 = line_intersection(h1, v1)
            p2 = line_intersection(h1, v2)
            p3 = line_intersection(h2, v1)
            p4 = line_intersection(h2, v2)
           

            #Confronto le x e le y dei punti per ricavare l'immagine 
            #più grande possibili così da comprendere sicuramente tutta la cella
            top = p1[1]
            if p1[1] < p2[1] :
                top = p2[1]
                if top < 0:
                    top = 0
            
            left = p1[0]
            if p1[0] < p3[0]:
                left = p3[0]
                if left < 0:
                    left = 0
            
            bottom = p3[1]
            if p3[1] > p4[1]:
                bottom = p4[1]
                if bottom < 0:
                    bottom = 0

            right = p2[0]
            if p2[0] > p4[0]:
                right = p4[0]
                if right < 0:
                    right = 0

            box = img[top : bottom, left : right]
            boxes.append(box)

    return boxes

### 4 found intersection of two lines
def line_intersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])


    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return int(x), int(y)


### 4 Print boxes
def printBoxes(boxes):
    dim = int(np.sqrt(len(boxes)))
    figure, axes = plt.subplots(nrows= dim, ncols= dim)

    k = 0
    for i in range(dim):
        for j in range(dim):
            axes[i][j].imshow(boxes[k])
            k = k+1

    plt.show()

### 4 GET PREDICTIONS ON ALL IMAGES
def getPredictions(boxes, model):
    print("I'm getting predictions...")
    result = []
    probs = []
    cut_value = 6
    i = 1
    for image in boxes:
        #Prepare Image
        img = image
        img = img[cut_value:img.shape[0] -cut_value , cut_value:img.shape[1]-cut_value] #Tolgo i bordi dell'immagine 
        img = preProcess(img)
        img = cv.resize(img, (28,28)) #ridimensioni in 28x28
        img = np.array([img])
        img = img.reshape(1,28,28,1)
        
        #immagine = img[0] #per il plot
        
        
        #Get Prediction
        prediction = model.predict(img)
        classIndex = np.argmax(prediction, axis = -1)  #ottengo la classe a cui con maggior probabilità il numero appartiene
        probValue = np.amax(prediction) #ottengo il valore di probabilità associato alla classe predetta
        #print(classIndex, probValue)

        result.append(classIndex[0])
        probs.append(probValue)
        
        
        # plt.title(f'Immagine {i}')
        # plt.imshow(immagine , cmap= plt.cm.gray)
        # plt.show()

       
        i = i+1
        

    return [result, probs] 


def coloraBordi(img):

    margin = 3

    for i in range(0, margin):
        for j in range(img.shape[1]):
            img[i][j] = 0



    for i in range(img.shape[0] - margin, img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 0



    for i in range(img.shape[0]):
        for j in range(0,margin):
            img[i][j] = 0



    for i in range(img.shape[0]):
        for j in range(img.shape[1]-margin, img.shape[1]):
            img[i][j] = 0



    return img




def correctGrid(grid, probs):
    dim = grid.shape[0]
    cell_with_ships = []
    index_founded = []

    for i in range(dim):
        for j in range(dim):

            orientation = ''
            good = True

            if grid[i][j] == 1 or grid[i][j] > 5 :
                grid[i][j] = 0

            if grid[i][j] != 0 and not ((i,j) in cell_with_ships):

                if grid[i][j] in index_founded:
                    good = False

                elif j == dim -1 : #Ultima colonna quindi guardo solo sotto
                    if i+1 < dim and grid[i+1][j] == grid[i][j]:
                        orientation = 'v'
                        isVertical = True
                        good = check(grid, i, j, isVertical, probs)
                    else:
                        good = False

                elif i == dim-1: #Ultima riga quindi guardo solo a destra
                    if j+1 < dim and grid[i][j+1] == grid[i][j]:
                        orientation = 'h'
                        isVertical = False
                        good = check(grid, i, j, isVertical, probs)
                    else:
                        good = False
                
                else:
                    if grid[i][j+1] != grid[i][j] and (i+1 < dim and grid[i+1][j] != grid[i][j]):
                        good = False
                    
                    elif i+1 < dim and grid[i+1][j] == grid[i][j]:
                        orientation = 'v'
                        isVertical = True
                        good = check(grid, i, j, isVertical, probs)
                    
                    elif j+1 < dim and grid[i][j+1] == grid[i][j]:
                        orientation = 'h'
                        isVertical = False
                        good = check(grid, i, j, isVertical, probs)
                
                if good:
                    bo = grid[i][j]
                    sr = i
                    sc = j
                    index_founded.append(grid[i][j])
                    if orientation == 'h':
                        for col in range(sc, sc+bo):
                            cell_with_ships.append((sr, col))
                    
                    if orientation == 'v':
                        for row in range(sr, sr+bo):
                            cell_with_ships.append((row, sc))
                else:
                    grid[i][j] = 0
            
            if (i == dim-1 and j == dim-1):
                if(grid[i-1][j] != grid[i][j] and grid[i][j-1] != grid[i][j]):
                    grid[i][j] = 0

    return grid






def check(grid, sr, sc, isVertical, probs):

    bo = grid[sr][sc]
    cell_to_check = grid[sr][sc]
    p = np.array([])
    
    if not isVertical:
        for col in range(sc, sc+bo):
            if col < grid.shape[1]:
                p = np.append(p, probs[sr][col])
                if grid[sr][col] != grid[sr][sc]:
                    return False
                cell_to_check = cell_to_check-1

    else:
        for row in range(sr, sr+bo):
            if row < grid.shape[0]:
                p = np.append(p, probs[row][sc])
                if grid[row][sc] != grid[sr][sc]:
                    return False
                cell_to_check = cell_to_check-1

    #Verifico se ci sono ancora celle da rimuovere
    if cell_to_check == 0:
        if isVertical:
            if sr+bo < grid.shape[1] and grid[sr+bo][sc] == grid[sr][sc] and probs[sr][sc] == p[0]:
                return False
        
        else:
            if sc+bo < grid.shape[0] and grid[sr][sc+bo] == grid[sr][sc] and probs[sr][sc] == p[0]:
                return False


        return True
    else:
        return False



                
                


    


    




