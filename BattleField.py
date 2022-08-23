import pygame, sys
import numpy as np
from random import randint, choice

from Grid import Grid
from Ship import Ship
from AI import *

#Questa classe gestisce il mainLoop 

class BattleField:

    #COLOR
    BG_COLOR = (28,170,156)
    LINE_COLOR = (23,145,135)

    BG_COLOR_P = (255,255,255)
    LINE_COLOR_P = (0,0,0)

    
    #TITLE
    TITLE = 'BATTLESHIP'
    TITLE_PROB = 'PROBABILITIES'

    #NumeroNavi
    

    def __init__(self, display_dim, gridConfig): 
        self.grid_dim_pixel = display_dim
        self.display_widht = display_dim*2+300
        self.display_height = display_dim
        self.gridConfig = gridConfig
        self.num_row_column = gridConfig.shape[0]
        

        self.ship = self.initializeShips(gridConfig)
        self.num_ships = self.ship.shape[0]


      

    def show(self):
        pygame.init()
        screen = pygame.display.set_mode(( self.display_widht, self.display_height))
        pygame.display.set_caption(self.TITLE)
        screen.fill(self.BG_COLOR)



        grid = Grid(self.display_widht, self.display_height, self.grid_dim_pixel, self.gridConfig)
        grid.draw_lines(screen, self.num_row_column)


        ai = AI(self.num_row_column, self.ship)
        [matrix_sum, maxProb] = ai.calcola_probabilita_iniziali()
        grid.setMaxProb(maxProb)
        grid.draw_probabilities(screen, matrix_sum)




        #mainloop
        while True:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    
                    #spara
                    [row, column] = ai.shot()
                    

                    if self.ship.size > 0:
                        if(grid.available_square(row, column)):
                            esito = grid.setHit(row,column)
                            ai.set_to_zero_hit_cell(row, column)
                            #aggiorna la nave colpita
                            if(esito == 'hit'):
                                ai.set_to_minus_one_cell(row, column)
                                for i in range(self.ship.size):
                                    sunk = self.ship[i].checkHit(self.ship[i].getIsVertical(), row, column)
                                    if (sunk):
                                        esito = 'sunk'
                                        ai.remove_matrix(self.ship[i])
                                        grid.set_hit_around_ship(self.ship[i])
                                        self.ship = np.delete(self.ship, i)
                                        
                                        break

                            
                            matrix_sum = ai.calcola_nuove_probabilita(row, column, esito)
                            grid.draw_figures(screen)
                            grid.draw_probabilities(screen, matrix_sum)
                            
                    else:
                        sys.exit()
                
            pygame.display.update()

                

    
    def initializeShips(self, gridConfig):
        
        dim = gridConfig.shape[0]
        water = 0
        index_founded = np.array([water])
        ship_array = np.array([])

        for i in range(dim):
            for j in range(dim):
                
                if not gridConfig[i][j] in index_founded:
                    index_founded = np.append(index_founded, gridConfig[i][j])
                    blockOccupied = int(gridConfig[i][j])
                    if( j+1 < dim and gridConfig[i][j+1] == gridConfig[i][j]):
                        #Nave in orizzontale
                        ship_array = np.append(ship_array, Ship(blockOccupied, i, j, False))
                    elif ( i+1 < dim and gridConfig[i+1][j] == gridConfig[i][j]):
                        #Nave in verticale   
                        ship_array = np.append(ship_array, Ship(blockOccupied, i, j, True))
        
        #print('Index_founded = ', index_founded)
        
        # print("Navi prima dell'ordinamento:")
        # for ship in ship_array:
        #     ship.to_String()

        # ship_array = self.sortShips(ship_array)

        # print("\n\nNavi dopo dell'ordinamento:")
        # for ship in ship_array:
        #     ship.to_String()

        return ship_array

    
    def sortShips(self, ship_array):
        for i in range(0, len(ship_array)):
            
            min = i
            ship_min = ship_array[i]
            
            for j in range(i+1, len(ship_array)):
                ship_j = ship_array[j]
                
                if ship_j.getTag() < ship_min.getTag():
                    min = j
                    ship_min = ship_j

            tmp = ship_array[i]
            ship_array[i] = ship_array[min]
            ship_array[min] = tmp
        
        return ship_array





       

    

        


        