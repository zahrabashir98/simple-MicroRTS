

import math
import numpy as np
import random
from random import choice

class Unit:
    def __init__(self, x, y, name, ownerID, hitpoints):
        """__init_ method"""
        self.x = x
        self.y = y
        self.name = name
        self.ownerID = ownerID
        self.hitpoints = hitpoints
        self.actionList = []
        self.active = False
        # self.pos = (self.x*4) + self.y

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
# heavy, light, worker for attacking
class meleeUnit(Unit):
    def canAttack(self, targetUnit):
        if self.x == targetUnit.x and abs(self.y - targetUnit.y) ==1:
            return True
        elif self.y == targetUnit.y and abs(self.x - targetUnit.x) ==1:
            return True
        else:
            return False

    def possibleAttackPos(self, targetUnit):
        chance = np.random.randint(2) 
        # print(chance)
        if chance ==0:
            y1 = targetUnit.y
            if targetUnit.x == 0:
                x1 = targetUnit.x + 1
                
            elif targetUnit.x == 1 or targetUnit.x == 2:
                chance2 = np.random.randint(2)  
                if chance2 ==0:
                    x1 = targetUnit.x - 1
                elif chance2 == 1:
                    x1 = targetUnit.x + 1
            
            elif targetUnit.x == 3:
                x1 = targetUnit.x - 1
     
        elif chance ==1:
            x1 = targetUnit.x
            if targetUnit.y ==0:
                y1 = targetUnit.y + 1
                
            elif targetUnit.y ==1 or targetUnit.y==2:
                chance3 = np.random.randint(2)
                if chance3 == 0:
                    y1 = targetUnit.y - 1
                elif chance3 == 1:
                    y1 = targetUnit.y + 1
            
            elif targetUnit.y ==3:
                y1 = targetUnit.y - 1

        return x1, y1
        

    def attackClosest(self, unitList):
        enemyUnits = []
        distances = []
        for unit in unitList:
            if unit.ownerID != self.ownerID:
                enemyUnits.append(unit)
                distances.append(math.sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        targetUnitIndex = distances.index(min(distances))
        targetUnit = enemyUnits[targetUnitIndex]
        # print(enemyUnits)
        # print(distances)
        # print(targetUnit.name)

        if self.canAttack(targetUnit):
            print("YES")
            self.actionList.append(["attack", targetUnit.name])

        else:
            print("NO")
            x1, y1 = self.possibleAttackPos(targetUnit) #TODO: find the shortest later / debug the passing over issue!
            # print("X1 X2\n")
            # print(x1, y1, x2, y2)
            diffX = x1 - self.x 
            diffY = y1 - self.y

            if diffX >0:
                for i in range(abs(diffX)):
                    self.actionList.append(["down", None])
            elif diffX<0:
                for i in range(abs(diffX)):
                    self.actionList.append(["up", None])
            
            if diffY >0:
                for i in range(abs(diffY)):
                    self.actionList.append(["right", None])
            elif diffY<0:
                for i in range(abs(diffY)):
                    self.actionList.append(["left", None])

            # findAttackPath()
            self.actionList.append(["attack", targetUnit.name])
            # self.actionList.append(["attack2", targetUnit])
        # print(self.actionList)

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

def countUnitTypes(UnitList, q):
    unitTypes = {"ranged":0, "heavy":0, "light":0, "worker":0, "barrack":0, "base":0}
    for each in UnitList:
        if isinstance(each, rangedUnit):
            unitTypes["ranged"] += 1
        elif isinstance(each, workerUnit):
            unitTypes["worker"] += 1
        elif isinstance(each, Barrack):
            unitTypes["barrack"] += 1
    return unitTypes[q]
                

class Barrack(Unit):
    def checkIfEmpty(self, pos, board):
        print("BOARD")
        print(board)
        if board[pos] != "  ":
            return False
        else:
            return True
    
    def getSoldierPosition(self, board): #THIS IS WRONG!
        x1 = np.random.randint(4)  
        y1 = np.random.randint(4)
        while(self.checkIfEmpty((4*x1 + y1), board)==False):
            x1 = np.random.randint(4)  
            y1 = np.random.randint(4) 

        return x1, y1


    def train(self, type, ownerID, hitpoints, unitList, board):
        if type =="ranged":
            num = countUnitTypes(unitList, "ranged")
            # input("CHOOSING the location for training ranged")
            x1, y1 = self.getSoldierPosition(board) #TODO: check if it can be placed there and it is not filled with other units
            print("LOCS for ranged", x1,y1)
            # input()
            self.actionList.append(["trainRanged", x1, y1, ownerID, hitpoints, num])


################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

class workerUnit(meleeUnit):
    def collect():
        pass

    def build(self, type, x, y, ownerID, hitpoints):
        if type == "barrack": #TODO: add exceptions for not being able to add a unit at a specific position becuase it is full!
            self.actionList.append(["buildBarrack", x, y, ownerID, hitpoints])

        elif type == "base":
            pass

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
class rangedUnit(Unit):
    def calculateDistance(self, x1, x2, y1, y2):
        return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

    def canAttack(self, targetUnit):
        if self.calculateDistance(self.x, targetUnit.x, self.y, targetUnit.y) <=2:
            return True
        else:
            return False

    def possibleAttackPos(self, targetUnit):
        chance = np.random.randint(2) 
        if chance ==0:
            y1 = targetUnit.y
            if targetUnit.x <=1:
                x1 = targetUnit.x + 2
                
            elif targetUnit.x >=2:
                x1 = targetUnit.x - 2
        elif chance ==1:
            x1 = targetUnit.x
            if targetUnit.y <=1:
                y1 = targetUnit.y + 2
                
            elif targetUnit.y >=2:
                y1 = targetUnit.y - 2

        return x1, y1
        
    def attackClosest(self, unitList):
        enemyUnits = []
        distances = []
        for unit in unitList:
            if unit.ownerID != self.ownerID:
                # print(unit.name)
                enemyUnits.append(unit)
                distances.append(math.sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        print(distances)
        if len(unitList) >1:
            targetUnitIndex = distances.index(min(distances))
            targetUnit = enemyUnits[targetUnitIndex]
            print(targetUnit.name)
            if self.canAttack(targetUnit):
                # print("YES")
                self.actionList.append(["attack", targetUnit.name]) # the last to be droppped and applied
                # self.actionList.append(["attack2", targetUnit])

            else:
                # print("NO")
                x1, y1 = self.possibleAttackPos(targetUnit) #TODO: find the shortest later / debug the passing over issue!
                diffX = x1 - self.x 
                diffY = y1 - self.y

                chance = np.random.randint(2) 
                if chance == 0:
                    print("CHANCE 0")
                    if diffX >0:
                        self.actionList.append(["down", None])
                    elif diffX<0:
                        self.actionList.append(["up", None])
                    elif diffX ==0:
                        if diffY >0:
                            self.actionList.append(["right", None])
                        elif diffY<0:
                            self.actionList.append(["left", None])
            
                elif chance == 1:
                    print("CHANCE 1")
                    if diffY >0:
                        self.actionList.append(["right", None])
                    elif diffY<0:
                        self.actionList.append(["left", None])
                    elif diffY ==0:
                        if diffX >0:
                            self.actionList.append(["down", None])
                        elif diffX<0:
                            self.actionList.append(["up", None])

            # self.actionList.append(["attack2", targetUnit])
        # print(self.actionList)
    #################################################################################################################################
    def ALLpossibleAttackPos(self, targetUnit):

        y1 = targetUnit.y
        if targetUnit.x <=1:
            x1 = targetUnit.x + 2
            
        elif targetUnit.x >=2:
            x1 = targetUnit.x - 2

        x2 = targetUnit.x
        if targetUnit.y <=1:
            y2 = targetUnit.y + 2
            
        elif targetUnit.y >=2:
            y2 = targetUnit.y - 2

        return x1, y1, x2, y2
    

    def allPossibleActions(self, unitList):
        Pool = ["left", "right", "up", "down"]
        indexes = [16, 17, 18, 19]

        if self.y ==0:
            Pool.remove("left")
            indexes.remove(16)
        if self.y ==3:
            Pool.remove("right")
            indexes.remove(17)
        if self.x ==0:
            Pool.remove("up")
            indexes.remove(18)
        if self.x ==3:
            Pool.remove("down")
            indexes.remove(19)
    
        enemyUnits = []
        distances = []
        for unit in unitList:
            if unit.ownerID != self.ownerID:
                # print(unit.name)
                enemyUnits.append(unit)
                distances.append(math.sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        
        # print(enemyUnits)
        if len(unitList) >1:
            targetUnitIndex = distances.index(min(distances))
            targetUnit = enemyUnits[targetUnitIndex]

            if self.canAttack(targetUnit):
                # print("ATTACK MIKONE")
                # print(targetUnit.name)
                Pool.append("attack_%s"%targetUnit.name)
                indexes.append(targetUnit.x*4 + targetUnit.y)

        # random.shuffle(Pool)
        # print(Pool)
        # input()
        
        return Pool, indexes
    
    def allPossibleAttacks(self, unitList):
        enemyUnits = []
        distances = []
        for unit in unitList:
            if unit.ownerID != self.ownerID:
                # print(unit.name)
                enemyUnits.append(unit)
                distances.append(math.sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        
        if len(unitList) >1:
            targetUnitIndex = distances.index(min(distances))
            targetUnit = enemyUnits[targetUnitIndex]

            allActions = []
            if self.canAttack(targetUnit):
                # print("YES")
                allActions.append("attack_%s"%targetUnit.name)
                # self.actionList.append(["attack", targetUnit]) 

            else:
                # print("NO")
                x1, y1, x2, y2 = self.ALLpossibleAttackPos(targetUnit)
                # print(x1, y1, x2, y2)
                diffX1 = x1 - self.x 
                diffY1 = y1 - self.y
                diffX2 = x2 - self.x 
                diffY2 = y2 - self.y
                

                if diffX1 >0:
                    allActions.append("down")
                elif diffX1<0:
                    allActions.append("up")
                
                if diffY1 >0:
                    allActions.append("right")
                elif diffY1<0:
                    allActions.append("left")

                if diffX2 >0:
                    allActions.append("down")
                elif diffX2<0:
                    allActions.append("up")
                
                if diffY2 >0:
                    allActions.append("right")
                elif diffY2<0:
                    allActions.append("left")
                
                # allActions.append("attack")
            random.shuffle(allActions)
            return allActions
        else:
            return ["none"]
            # self.actionList.append(["attack", targetUnit])
