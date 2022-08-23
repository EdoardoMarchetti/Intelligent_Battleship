class Node:

    def __init__(self, row, column, prob):
        self.row = row
        self.column = column
        self.prob = prob

    def set_prob(self, newProb):
        self.prob = newProb

    def get_prob(self):
        return self.prob

    def get_coordinates(self):
        return [self.row, self.column]

    def get_row(self):
        return self.row
    
    def get_column(self):
        return self.column
    
    def to_string(self):
        print("Cella: R-", self.row, " || C- ", self.column," || P- ", self.prob)