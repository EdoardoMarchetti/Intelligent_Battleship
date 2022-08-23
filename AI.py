from random import randint
import numpy as np
from Node import Node

class AI:

    #Lista matrici delle navi (una per le configurazioni orizzontali e una per le configurazioni verticali)
    

    def __init__(self, num_row_column, shipArray):
        self.matrix_sum = np.zeros((num_row_column, num_row_column))    #matrice somma
        self.num_row_column = num_row_column                            #dimensione matrice
        self.num_ships = len(shipArray)                                      #numero navi iniziale
        self.ship_array = shipArray
        self.crea_matrici()                                     
        self.reference_coordinates = np.array([-1,-1])                  #coordinate di riferimento quando si colpisce per la prima volta
        self.frontier = np.array([])                                    #Lista frontier
        self.explored = []


    def crea_matrici(self):
        self.matrix_list = np.empty((0,3)) #3 colonne = , indice nave, matrice_o, matrice_v
 
        for ship in self.ship_array:
            num_nave = ship.getTag()       #Poichè l'identificativo della nave è il numero di blocchi occupati
            matrice_o = np.zeros((self.num_row_column , self.num_row_column ), int)
            matrice_v = np.zeros((self.num_row_column , self.num_row_column ), int)
            
            self.matrix_list  = np.append(self.matrix_list , np.array([[ num_nave, matrice_o, matrice_v]]), axis=0) 


    def calcola_probabilita_iniziali(self):

        #Ciclo per scorrere le navi
        i = 0 
        for ship in self.ship_array:
            dim_ship = ship.getTag()
            matrice_o = self.matrix_list[i][1]

            #Ciclo per inizializzare l'orizzontale
            for j in range(self.num_row_column):
                for k in range(self.num_row_column):
                    if(k+dim_ship-1 < self.num_row_column):
                        z = 0
                        while z < dim_ship:
                            matrice_o[j][k+z] = matrice_o[j][k+z] + 1
                            z = z+1

            self.matrix_list[i][1] = matrice_o


            #Ciclo per inizializzare verticale
            matrice_v = self.matrix_list[i][2]
            for j in range(self.num_row_column):
                for k in range(self.num_row_column):
                    if(j+dim_ship-1 < self.num_row_column):
                        z = 0
                        while z < dim_ship:
                            matrice_v[j+z][k] = matrice_v[j+z][k] + 1
                            z = z+1

            self.matrix_list[i][2] = matrice_v

            [matrix_sum, max] = self.aggiorna_matrice_somma()
            i = i+1
        
        
        return [matrix_sum, max]

    
    def aggiorna_matrice_somma(self):
        self.matrix_sum = np.zeros((self.num_row_column, self.num_row_column))

        for i in range(self.matrix_list.shape[0]):
            self.matrix_sum = self.matrix_sum + self.matrix_list[i][1]
            self.matrix_sum = self.matrix_sum + self.matrix_list[i][2]

        #self.stampa_matrice_somma()

        #Ricerca del massimo 
        max = 0.0
        for i in range(self.matrix_sum.shape[0]):
            for j in range(self.matrix_sum.shape[1]):
                if self.matrix_sum[i][j] > max:
                    max = self.matrix_sum[i][j]

        return [self.matrix_sum, max]



    def shot(self):
        if(not self.frontier.size == 0): #Devo scegliere in base a dove ho colpito
            [row, column] = self.frontier[0].get_coordinates()
            return [row, column]
        else:
            max = np.array([-1,-1, -1]) #[riga,colonna,probabilità]
            for j in range(self.num_row_column):
                    for k in range(self.num_row_column):
                        if(max[2] < self.matrix_sum[j][k]):
                            max = [j,k, self.matrix_sum[j][k]]

            return [max[0], max[1]]

    

    def set_to_zero_hit_cell(self, hit_row, hit_column):
        for i in range(self.matrix_list.shape[0]):
            matrice_o = self.matrix_list[i][1]
            matrice_v = self.matrix_list[i][2]

            matrice_o[hit_row][hit_column] = 0
            matrice_v[hit_row][hit_column] = 0

            self.matrix_list[i][1] = matrice_o
            self.matrix_list[i][2] = matrice_v
            

            self.aggiorna_matrice_somma()
    
    def set_to_minus_one_cell(self, hit_row, hit_column):
        for i in range(self.matrix_list.shape[0]):
            matrice_o = self.matrix_list[i][1]
            matrice_v = self.matrix_list[i][2]

            matrice_o[hit_row][hit_column] = -1
            matrice_v[hit_row][hit_column] = -1

            self.matrix_list[i][1] = matrice_o
            self.matrix_list[i][2] = matrice_v
            

            self.aggiorna_matrice_somma()

    


    
    def remove_matrix(self, sunk_ship):
        

        #Rimuovo la riga associata alla nave affondata
        removed = False
        i = 0
        while i < self.matrix_list.shape[0] and (not removed):
            if (self.matrix_list[i][0] == sunk_ship.getTag()):
                self.matrix_list = np.delete(self.matrix_list, i, 0)
                removed = True
            i = i+1

        #Caratteristiche nave affondata
        sr = sunk_ship.getStartRow()
        sc = sunk_ship.getStartColumn()
        bo = sunk_ship.getTag()

        #Aggiorno le matrici delle altre navi poichè non possono stare attacate
        #Simulo dei colpi con esito WATER intorno alla nave

        if(sunk_ship.getIsVertical()):
                #Nave affondata in verticale
                for row in range(sr-1, sr+bo+1):
                    if(row >= 0 and row < self.num_row_column):
                        for column in range (sc-1, sc+2):
                            if(column >= 0 and column<self.num_row_column and self.matrix_sum[row][column] > 0):
                                self.set_to_zero_hit_cell(row,column)
                                self.aggiorna_matrici(row,column)
            
        else:
            #Nave affondata in orizzontale
            for row in range(sr-1, sr+2):
                if (row >= 0 and row < self.num_row_column):
                    for column in range (sc-1, sc+bo+1):
                        if(column >= 0 and column<self.num_row_column and self.matrix_sum[row][column] > 0):
                            self.set_to_zero_hit_cell(row,column)
                            self.aggiorna_matrici(row,column)



    def calcola_nuove_probabilita(self, hit_row, hit_column, esito):

        if(esito == 'water'):
            self.aggiorna_matrici(hit_row, hit_column)
            

        elif(esito == 'hit'):
            #Se non ho agganciato ancora nessuna nave salvo dove l'ho trovata
            if(self.reference_coordinates[0] == -1 and self.reference_coordinates[1] == -1):
                self.reference_coordinates = [hit_row, hit_column]
            
            self.aggiorna_frontier(hit_row, hit_column)
        
        elif(esito == 'sunk'):
            self.frontier = np.array([])
            self.reference_coordinates = np.array([-1, -1])


        self.aggiorna_matrice_somma()
        self.remove_node_from_frontier(hit_row, hit_column)
        return  self.matrix_sum


    

    def aggiorna_matrici(self, hit_row, hit_column):

        #Scadisco la lista delle matrici nave per nave
        for i in range(int (self.matrix_list.shape[0])):
            
            blockOccupied = self.matrix_list[i][0] #indice nave = blocchi occupati
            matrice_o = self.matrix_list[i][1] 
            matrice_v = self.matrix_list[i][2]

            #Aggiorno orizzontali
            for j in range(hit_column - blockOccupied +1 , hit_column +1):
                if(j>=0 and j+blockOccupied-1 < self.num_row_column and (not (hit_row, j) in self.explored) ): 
                    good = True

                    for k in range(blockOccupied):
                        if(j+k != hit_column and matrice_o[hit_row][j+k] == 0 ): 
                            good = False
                    
                    if good:
                        for k in range(blockOccupied):
                            if(matrice_o[hit_row][j+k] > 0): 
                                matrice_o[hit_row][j+k] = matrice_o[hit_row][j+k] - 1


            #Aggiorno verticali
            for j in range(hit_row-blockOccupied+1 , hit_row+1):
                if(j>=0 and j+blockOccupied-1 < self.num_row_column and (not (j, hit_column) in self.explored)):
                    good = True
                    for k in range(blockOccupied):
                        if(j+k != hit_row and matrice_v[j+k][hit_column] == 0 ): 
                            good = False
                    
                    if good:
                        for k in range(blockOccupied):
                            if(matrice_v[j+k][hit_column] > 0):
                                matrice_v[j+k][hit_column] = matrice_v[j+k][hit_column] - 1
                                


            self.matrix_list[i][1] = matrice_o
            self.matrix_list[i][2] = matrice_v
       
        self.explored.append((hit_row, hit_column))



     
    
    


    def aggiorna_frontier(self, hit_row, hit_column):
        
        if(self.frontier.size == 0): #Aggiungo le caselle adiacenti con prob != 0
                #Sopra
                if (hit_row - 1 >= 0 and self.matrix_sum[hit_row-1][hit_column] > 0 ):
                    self.frontier = np.append(self.frontier,  Node(hit_row-1, hit_column, self.matrix_sum[hit_row-1][hit_column]))
                    self.printFrontier()
                
                #Sotto
                if(hit_row + 1 < self.num_row_column and self.matrix_sum[hit_row+1][hit_column] > 0):
                    self.frontier = np.append(self.frontier, Node(hit_row+1, hit_column, self.matrix_sum[hit_row+1][hit_column]))
                    self.printFrontier()

                #Destra
                if(hit_column + 1 < self.num_row_column and self.matrix_sum[hit_row][hit_column+1] > 0):
                    self.frontier = np.append(self.frontier, Node(hit_row, hit_column+1, self.matrix_sum[hit_row][hit_column+1]))
                    self.printFrontier()
                
                #Sinistra
                if(hit_column-1 >= 0 and self.matrix_sum[hit_row][hit_column-1] > 0):
                    self.frontier = np.append(self.frontier, Node(hit_row, hit_column-1, self.matrix_sum[hit_row][hit_column-1])) 
                    self.printFrontier() 
        
        else: 
            #Aggiungo le caselle adiacenti a quella che ho colpito con prob != 0 e con hit_row == reference_row (oppure con hit_column == reference_column)

            #verifico se hit_row == reference row 
            if (hit_row == self.reference_coordinates[0]):
                
                #Se SI allora la nave è in orizzontale

                #Tolgo i nodi con row != reference_row
                self.remove_node_from_frontier(self.reference_coordinates[0] - 1, self.reference_coordinates[1])
                self.remove_node_from_frontier(self.reference_coordinates[0] + 1, self.reference_coordinates[1])

                #Aggiungo i nodi adiacenti sulla stessa riga e con prob != 0
                #Destra
                if(hit_column + 1 < self.num_row_column and self.matrix_sum[hit_row][hit_column+1] > 0):
                    self.frontier = np.append(self.frontier, Node(hit_row, hit_column+1, self.matrix_sum[hit_row][hit_column+1]))
                    self.printFrontier()
                
                #Sinistra
                if(hit_column-1 >= 0 and self.matrix_sum[hit_row][hit_column-1] > 0):
                    self.frontier = np.append(self.frontier, Node(hit_row, hit_column-1, self.matrix_sum[hit_row][hit_column-1])) 
                    self.printFrontier() 
            
            else:
                #Se NO allora nave in verticale
                #Tolgo i nodi con column != reference_column
                self.remove_node_from_frontier(self.reference_coordinates[0], self.reference_coordinates[1] - 1)
                self.remove_node_from_frontier(self.reference_coordinates[0], self.reference_coordinates[1] + 1)

                #Aggiungo i nodi adiacenti sulla stessa riga e con prob != 0
                #Sopra
                if (hit_row - 1 >= 0 and self.matrix_sum[hit_row-1][hit_column] > 0 ):
                    self.frontier = np.append(self.frontier,  Node(hit_row-1, hit_column, self.matrix_sum[hit_row-1][hit_column]))
                    self.printFrontier()
                
                #Sotto
                if(hit_row + 1 < self.num_row_column and self.matrix_sum[hit_row+1][hit_column] > 0):
                    self.frontier = np.append(self.frontier, Node(hit_row+1, hit_column, self.matrix_sum[hit_row+1][hit_column]))
                    self.printFrontier()



    def remove_node_from_frontier(self, row_to_remove ,column_to_remove):
        #Devo rimuovere il nodo con coordinate passate
        i = 0
        removed = False
        while (i < self.frontier.size and (not removed)):
            node = self.frontier[i]
            if node.get_row() == row_to_remove and node.get_column() == column_to_remove:
                self.frontier = np.delete(self.frontier, i)
                removed = True
            i = i+1
        
        self.printFrontier()

        
        




    """........................Metodi di utilità.........................."""
    def stampa_matrice_somma(self):
        print("Matrice somma = \n",self.matrix_sum)

    def printFrontier(self):
        print("Size: " , self.frontier.size)
        for i in range (self.frontier.size):
               self.frontier[i].to_string()

    
    

    
    