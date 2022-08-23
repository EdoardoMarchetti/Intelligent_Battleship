
from BattleField import BattleField
import numpy as np


gridConfig = np.array([
    [0,0,0,0,0],
    [0,3,3,3,0],
    [0,0,0,0,0],
    [2,0,0,0,0],
    [2,0,0,0,0]
    
])


bf = BattleField(500, gridConfig)
bf.show()


