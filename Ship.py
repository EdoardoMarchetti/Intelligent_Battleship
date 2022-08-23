import numpy as np

class Ship:

    def __init__(self, lives, startRow, startColumn, isVertical):
        self.tag = lives
        self.lives = lives
        self.startRow = startRow
        self.startColumn = startColumn
        self.isVertical = isVertical

    
    def checkHit(self, isVertical, row, column):
        if(isVertical and row>= self.startRow  and row < self.startRow + self.tag and column == self.startColumn):
            self.lives = self.lives - 1

        elif(not isVertical and column >= self.startColumn and column < self.startColumn+self.tag and row == self.startRow):
            self.lives = self.lives - 1
        
        if self.lives == 0: 
            return True
        return False

    def getTag(self):
        return self.tag

    def getIsVertical(self):
        return self.isVertical

    def getStartRow(self):
        return self.startRow

    def getStartColumn(self):
        return self.startColumn

    def to_String(self):
        print("Tag nave = ", self.tag,
                "\nStart_row = ", self.startRow,
                "\nStart_column = ", self.startColumn,
                "\nIsVertical = ", self.isVertical)