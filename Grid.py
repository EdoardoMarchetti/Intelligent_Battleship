import pygame, sys
import numpy as np

#Questa classe si occupa di diseganre la griglia e di tener traccia dei colpi effettuati

class Grid:

    #GRAPHICS ITEM
    BLACK = (0,0,0)
    RED = (255, 0, 0)
    LINE_COLOR = (23,145,135)
    CIRCLE_STROKE = 5
    CROSS_STROKE = 5

    #SYMBOLS
    NOT_HIT_WATER = 0 #'0' #Con Battlefield originale mettere '0'
    WATER = -1 # Acqua colpita
    SHIP = -2 # Nave colpita


    def __init__(self, display_w, display_h, grid_dim_pixel, gridConfiguration):
        
        self.display_w = display_w
        self.display_h = display_h

        self.board = gridConfiguration
        self.num_row = gridConfiguration.shape[0]
        self.num_col = gridConfiguration.shape[1]
        self.grid_dim_pixel = grid_dim_pixel
        self.cell_dim = grid_dim_pixel / self.num_row

        self.circle_radius = (self.cell_dim - 10) / 2

        self.stroke = 5
        

        

    def initialize_board(self):
        self.board = []
        for row in range(self.num_row):
            self.board.append([self.NOT_HIT_WATER for x in range(self.num_col)])
                

    def draw_lines(self, screen, numLines):
        
        for i in range(1,numLines+1):
            #vertical
            pygame.draw.line(screen, self.LINE_COLOR, (self.cell_dim * i, 0) , (self.cell_dim * i , self.display_h), self.stroke) #da sinistra verso destra
            pygame.draw.line(screen, self.LINE_COLOR, (self.display_w - (self.cell_dim * i), 0), (self.display_w - (self.cell_dim * i), self.display_h), self.stroke) #da destra verso sinistra                                                                                                #da destra verso sinistra  
            #horizontal
            pygame.draw.line (screen, self.LINE_COLOR, (0, self.cell_dim * i), (self.grid_dim_pixel, self.cell_dim * i), self.stroke)
            pygame.draw.line (screen, self.LINE_COLOR, (self.display_w - self.grid_dim_pixel, self.cell_dim * i), (self.display_w, self.cell_dim * i), self.stroke)


    def mark_square(self, row, col, player):
        self.board[row][col] = player
 
    


    def available_square(self, row, col):
        return ((not self.board[row][col] == self.WATER) and (not self.board[row][col] == self.SHIP))
    
    def available_square_for_posiotioning(self, row, col):
        return self.board[row][col] == self.NOT_HIT_WATER

    
    def draw_figures(self, screen):
        for row in range(self.num_row):
            for col in range(self.num_col):

                if self.board[row][col] == self.SHIP:
                    pygame.draw.circle(screen, self.RED, (int(col * self.cell_dim + self.cell_dim / 2) , int(row * self.cell_dim + self.cell_dim / 2)), self.circle_radius, self.CIRCLE_STROKE) 

                if self.board[row][col] == self.WATER:  
                    pygame.draw.line(screen, self.BLACK, (col * self.cell_dim, row * self.cell_dim), ((col+1) * self.cell_dim, (row+1) * self.cell_dim), self.CROSS_STROKE)  
                    pygame.draw.line(screen, self.BLACK, (col * self.cell_dim, (row+1) * self.cell_dim), ((col+1) * self.cell_dim, (row) * self.cell_dim), self.CROSS_STROKE)  

                  
  

    
    def print_board(self):
        for row in range(self.num_row):
            print(self.board[row])

    
    
    def positionShip(self, startRow, startColumn, orientation, blockOccupied):
        if(orientation):
            for i in range(blockOccupied):
                self.mark_square(startRow+i, startColumn, blockOccupied)
        else:
            for i in range(blockOccupied):
                self.mark_square(startRow, startColumn+i, blockOccupied)

    
    def setHit(self, row, column):
        if(self.board[row][column] == self.NOT_HIT_WATER):
            print(self.board[row][column])
            self.mark_square(row, column, self.WATER)
            return 'water'
           
        else:
            print(self.board[row][column])
            self.mark_square(row, column, self.SHIP)
            return 'hit'

    
    def set_hit_around_ship(self, ship):
        sr = ship.getStartRow()
        sc = ship.getStartColumn()
        bo = ship.getTag()     

        if(ship.getIsVertical()):
                #Nave affondata in verticale
                for row in range(sr-1, sr+bo+1):
                    if(row >= 0 and row < self.num_row):
                        for column in range (sc-1, sc+2):
                            if(column >= 0 and column<self.num_col and self.available_square(row,column)):
                                self.setHit(row,column)
            
        else:
            #Nave affondata in orizzontale
            for row in range(sr-1, sr+2):
                if (row >= 0 and row < self.num_row):
                    for column in range (sc-1, sc+bo+1):
                        if(column >= 0 and column<self.num_col and self.available_square(row,column)):
                            self.setHit(row,column)
    
    
    
    def setMaxProb(self, maxProb):
        self.max_prob = maxProb
        

    
    def draw_probabilities(self, screen, matrix_sum):
       
        start_horizontal = self.display_w - self.grid_dim_pixel
        color_sample = (255, 0, 0) / self.max_prob
        
        
        for i in range(self.num_row):
            for j in range(self.num_col):
                rect = pygame.Rect(start_horizontal + self.cell_dim *j, self.cell_dim * i, self.cell_dim, self.cell_dim)
                
                coeff = matrix_sum[i][j]

                if matrix_sum[i][j] <= 0:
                    coeff = 0
                
                color = color_sample * coeff
                pygame.draw.rect(screen, color, rect, 0)
                




