import math
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from random import choice
import itertools
import copy
import random
from collections import namedtuple
from MCTS import MCTS, Node
from NN import *

np.random.seed(1)
random.seed(1)
class Board():

    def __init__(self):
        """__init__ method"""
        self.x = "-"
        self.y = "|"
        self.z = "  "
        self.zz = ""
        self.board = ["  "]*16
        self.number_of_units = 2
        self.unitList = []
        self.playerBasedList = [[],[]]
        self.is_over = False
        self.winner = 0 #no one has won yet, 1: player 1, 2: player 2, 0.5: tie
        self.p1_actions = []
        self.p2_actions = []
        self.nodeType = "max" #TODO: is this correct?

    
    def hash(self): #TODO: CHANGE FOR multi-units game
        p1 = []
        p2 = []
        if self.p1_actions != []:
            p1 = self.p1_actions[0]
        if self.p2_actions != []:
            p2 = self.p2_actions[0]
        u2_hit = 0
        u1_hit = 0
        for each in self.unitList:
            if each.ownerID == 1:
                u1_hit = each.hitpoints
            elif each.ownerID == 2:
                u2_hit = each.hitpoints

        return (tuple(self.board), self.nodeType, tuple(p1), tuple(p2), u1_hit, u2_hit, self.winner, self.is_over)
    
    # def findEquivalentBoard(self):
    #     return equivalentBoard(self.board, self.nodeType, self.winner, self.is_over)


    def Draw(self):
        """Draws the board with square numbers"""
        print ('The Board with the square numbers looks like this: \n')     # draw the borad with the square numbers
        print ((self.z * 10), self.board[0], self.zz, self.y, self.board[1], self.zz, self.y, self.board[2], self.zz, self.y, self.board[3])
        print ((self.z * 10), self.x * 22)
        print ((self.z * 10), self.board[4], self.zz, self.y, self.board[5], self.zz, self.y, self.board[6], self.zz, self.y, self.board[7])
        print ((self.z * 10), self.x * 22)
        print ((self.z * 10), self.board[8], self.zz, self.y, self.board[9], self.zz, self.y, self.board[10], self.zz, self.y, self.board[11])
        print ((self.z * 10), self.x * 22)
        print ((self.z * 10), self.board[12], self.zz, self.y, self.board[13], self.zz, self.y, self.board[14], self.zz, self.y, self.board[15])
        print ('\n')

    def update_board(self):
        self.board = ["  "]*16
        remove_list = []
        for unit in self.unitList:
            if unit.hitpoints >=0:
                self.board[(unit.x*4) + unit.y] = unit.name 
            elif unit.hitpoints <0:
                remove_list.append(unit)
        
        for _ in range(len(remove_list)):
            for unit in self.unitList:
                if unit in remove_list:
                    # print("SAME")
                    self.unitList.remove(unit)
                    break
    
    def updatePlayerBasedList(self):
        self.playerBasedList = [[],[]]
        for each in self.unitList:
            if each.ownerID == 1:    
                self.playerBasedList[0].append(each)
            elif each.ownerID == 2:    
                self.playerBasedList[1].append(each)

    def isOver(self): #TODO: optimize this

        player1 = self.playerBasedList[0]
        player2 = self.playerBasedList[1]
        
        num_p1 = 0
        num_p2 = 0
        for item in player1:
            if item.hitpoints <=0:
                num_p1 += 1
    
        for item in player2:
            if item.hitpoints <=0:
                num_p2 += 1
        if num_p1 == len(player1) and num_p2 == len(player2):
            self.winner = 0.5
            return True
        elif num_p1 == len(player1):
            self.winner = 2
            return True
        elif num_p2 == len(player2):
            self.winner = 1
            return True

        if len(self.unitList) <=1:
            return True
        
        for unit in player1:
            if unit.hitpoints <= 0:
                continue
            else:
                break
            
        for unit in player2:
            if unit.hitpoints <= 0:
                continue
            else:
                return False

        return True
    
    
    def find_children(self, P):
        if self.isOver():  # If the game is finished then no moves can be made
            return [[],[], []] #set()
        # assert self.nodeType == player
        allActions = []
        if self.nodeType == "max":
            for each in self.unitList:
                if isinstance(each, rangedUnit): #TODO: support other types later
                    if each.ownerID == 2:
                        allActions = set(each.allPossibleAttacks(self.unitList))
        elif self.nodeType == "min":
            for each in self.unitList:
                if isinstance(each, rangedUnit): #TODO: support other types later
                    if each.ownerID == 1:
                        allActions = set(each.allPossibleActions(self.unitList))


        allObjs = []
        allHash = []
        allIndexes = []
        for action in allActions:
            obj, hassh, index = self.make_move(action)
            allObjs.append(obj)
            allHash.append(hassh)
            allIndexes.append(index)
 
        return allObjs, allHash, allIndexes

    def find_random_child(self):
        if self.is_over==True or self.isOver():
            print("IS OVER RANDOM CHILD")
            return None  # If the game is finished then no moves can be made

        childrenObjs, childrenHash = self.find_children()
        c = choice(childrenObjs)

        return c

    def returnUnitbyName(self, unitList, name): #TODO: make sure we don't have duplicate names
        for u in unitList:
            if u.name == name:
                return u
        
    def make_move(self, action): # TODO: remove duplicate states? (check every attribute)
        #TODO: this is for only 2 unit games. we have to extend it to other for than two unit ones later?
        
        # print("*********************START OF CHILDREN\n")
        if self.nodeType == "max":
            tmp = copy.deepcopy(self)
            tmp.nodeType = "min"
            tmp.p2_actions = []
            tmp.p1_actions = []
            for eachU in tmp.unitList:
                if eachU.ownerID == 2:
                    if action == "right" and eachU.y!=3:
                        tmp.p2_actions.append([action, None])
                        index = 17
                    elif action == "left" and eachU.y!=0:
                        tmp.p2_actions.append([action, None])
                        index = 16
                    elif action == "up" and eachU.x!=0:
                        tmp.p2_actions.append([action, None])
                        index = 18
                    elif action == "down" and eachU.x!=3:
                        tmp.p2_actions.append([action, None])
                        index = 19

                    elif action[:6] == "attack":
                        unitName = action[7:]
                        targetUnit = tmp.returnUnitbyName(tmp.unitList, unitName)
                        tmp.p2_actions.append([action[:6], targetUnit.name])
                        index = targetUnit.x * 4 + targetUnit.y

        elif self.nodeType == "min":

            tmp = copy.deepcopy(self)
            tmp.nodeType = "max"
            tmp.p1_actions = []
            tmp.p2_actions = self.p2_actions #get it from your father
            # print("MAKE MOVE< NODE MIN - prev_p2_actions")
            # print("ACTION", action)

            for eachU in tmp.unitList:
                if eachU.ownerID == 2:
                    if self.p2_actions[0][0] == "right" and eachU.y!=3:
                        eachU.y += 1
                    elif self.p2_actions[0][0]== "left" and eachU.y!=0:
                        eachU.y -= 1
                    elif self.p2_actions[0][0] == "up" and eachU.x!=0:
                        eachU.x -= 1
                    elif self.p2_actions[0][0] == "down" and eachU.x!=3:
                        eachU.x += 1

                    elif self.p2_actions[0][0] == "attack":
                        unitName = self.p2_actions[0][1]
                        targetUnit = tmp.returnUnitbyName(tmp.unitList, unitName)
                        if eachU.canAttack(targetUnit):
                            targetUnit.hitpoints -= 1
                        
            
                elif eachU.ownerID == 1:
                    if action == "right" and eachU.y!=3:
                        eachU.y += 1
                        index = 17
                        tmp.p1_actions.append([action, None])
                    elif action == "left" and eachU.y!=0:
                        eachU.y -= 1
                        index = 16
                        tmp.p1_actions.append([action, None])
                    elif action == "up" and eachU.x!=0:
                        eachU.x -= 1
                        index = 18
                        tmp.p1_actions.append([action, None])
                    elif action == "down" and eachU.x!=3:
                        eachU.x += 1
                        index = 19
                        tmp.p1_actions.append([action, None])
                    elif action[:6] == "attack":
                        unitName = action[7:]
                        targetUnit = tmp.returnUnitbyName(tmp.unitList, unitName)
                        tmp.p1_actions.append([action[:6], targetUnit.name])
                        if eachU.canAttack(targetUnit):
                            targetUnit.hitpoints -= 1
                        index = targetUnit.x * 4 + targetUnit.y
            # print(tmp.p1_actions)        
            tmp.update_board()
            tmp.updatePlayerBasedList()
        # tmp.Draw()
        
        if tmp.unitList:
            pass

        else:
            tmp.is_over = True
            # print("game is over")
        if tmp.isOver():
            tmp.is_over = True

        return tmp, tmp.hash(), index

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
            self.actionList.append(["attack", targetUnit])

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
            self.actionList.append(["attack", targetUnit])
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
                print("YES")
                self.actionList.append(["attack", targetUnit]) # the last to be droppped and applied
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
        if self.y ==0:
            Pool.remove("left")
        if self.y ==3:
            Pool.remove("right")
        if self.x ==0:
            Pool.remove("up")
        if self.x ==3:
            Pool.remove("down")
    
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

            if self.canAttack(targetUnit):
                Pool.append("attack_%s"%targetUnit.name)

        random.shuffle(Pool)
        return Pool


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



def evaluate(action, u_me, b, board):

    if action[0] == "attack":
        if isinstance(action[1], Unit):
            u_enemy = action[1]
        else:
            u_enemy = b.returnUnitbyName(b.unitList, action[1])
        # print(u_enemy.name)
        if u_me.canAttack(u_enemy):
            u_enemy.hitpoints -= 1 #TODO: remove the unit from the screen or don't let the hitpoint be less than 0; #DONE
            print("ATTACK to %s DONE"%u_enemy.name)
        else:
            print("attack not done")

    elif action[0] == "buildBarrack":
        #TODO: double check if it is empty to make; #DONE
        u = Barrack(action[1], action[2], "B%s"%action[3], action[3], action[4])
        b.unitList.append(u)


    elif action[0] == "trainRanged":
        #TODO: double check if it is empty to make; #DONE
        u = rangedUnit(action[1], action[2], "R%s"%(action[5]+2), action[3], action[4])
        b.unitList.append(u)  

    else:
        action = action[0] # direction, None, None
        # print(action)
        if action == "right":
            if u_me.y+1<= 3 and board[(u_me.x*4) + (u_me.y+1)] == "  ":
                u_me.y += 1
            else: 
                print("COULN'T MOVE")
    
        elif action == "left":
            if u_me.y-1>=0 and board[(u_me.x*4) + (u_me.y-1)] == "  ":
                u_me.y -= 1
            else: 
                print("COULN'T MOVE")
        elif action == "up":
            if u_me.x-1>=0 and board[((u_me.x -1)*4) + (u_me.y)] == "  ":
                u_me.x -= 1
            else: 
                print("COULN'T MOVE")
        elif action == "down":
            if u_me.x+1<=3 and board[((u_me.x +1)*4) + (u_me.y)] == "  ":
                u_me.x += 1
            else: 
                print("COULN'T MOVE")
        elif action == "nothing":
            print("ACTION: WAIT")


def create_dataset(board):
    data = np.zeros((4,4, 2))
    for eachU in board.unitList:
        if eachU.ownerID == 2:
            data[eachU.x][eachU.y] = [1, 0]
        else:
            data[eachU.x][eachU.y] = [0, 1]
    return data
    
###############################################################################################################
#                                                                                                             #
#                                                    Main                                                     #
#                                                                                                             #
###############################################################################################################
# assumption: I'm assuming all actions' durations are equal to one time step!

def start(x1, y1, x2, y2,x3, y3, max_iterations, model):
    b = Board()
    b.Draw()

    u2 = rangedUnit(x1, y1, "R2", 2, 4)
    u5 = rangedUnit(x2, y2, "R1", 1, 10)
    # u6 = rangedUnit(x3, y3, "R3", 1, 10)

    b.unitList.append(u2)
    b.unitList.append(u5)
    # b.unitList.append(u6)

    print(b.unitList)
    b.updatePlayerBasedList()
    print(b.playerBasedList)
    b.update_board()
    b.Draw()
    print(b.unitList)
    # input()

    gameOver = False
    t = 0
    dataset = []
    labels = []
    count = 0




    # model.fit(x_train, y_train_classification, epochs=10, batch_size=32)


    tree = MCTS()  #??
    memory_states  = []
    memory_pi = []
    memory_z = []
    # one game -> to the end
    root = b
    print(root.hash())
    input("THIS WAS ROOT")

    while(gameOver==False):
        for u in b.unitList:
            if u ==u2: # the player we are intertsed to learn its behaviour! (Ranged unit for now, and the behaviour is "attack closeset" for now!)
                # HERE WE CALL ALPHA ZERO
                memory_states.append(b)
                tree.Pi[b.hash()] = 1 #assigning root to 1
                for j in range(5):
                    tree.do_rollout(b, model)
                pi = tree.returnDist(b)
                memory_pi.append(pi)
                b = tree.choose(b)

            else:
                memory_states.append(b)
                tree.Pi[b.hash()] = 1 #assigning root to 1
                for j in range(5):
                    tree.do_rollout(b, model)
                pi = tree.returnDist(b)
                memory_pi.append(pi)
                b = tree.choose(b)


        # print(u5.actionList)
        # print(u2.actionList)

        # print("\nLET'S EVALUATE\n\n")

        # evaluate
        # for eachU in b.unitList:
        #     if eachU.actionList:
        #         action1 = eachU.actionList.pop(0)
        #         print("FIRST ACTION of %s POPPED: "%eachU.name)
        #         if len(eachU.actionList) != 0:
        #             AssertionError
        #         print(action1)
        #         evaluate(action1, eachU, b, b.board)
        print(b.p1_actions)
        print(b.p2_actions)
        b.updatePlayerBasedList()
        b.update_board()
        b.Draw()
        # input("NEXT FIND CHILDREN for the next play")
 
        if b.isOver():
            gameOver =True
            b.is_over = True

        for u in b.unitList:
            print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

        # print(t)
        input("T=%s ended!"%t)
        
        t +=1
        if t>=100:
            input("MORE THAN 100, exit")
            b.is_over = True
            b.winner = 0.5
            break
 
    input("biroone while\n")
    if root.nodeType == "max" and b.winner == 2:
        val = 1
        for _ in range(len(memory_states)):
            memory_z.append(val)
            val *= -1
    elif root.nodeType == "max" and b.winner == 1:
        val = -1
        for _ in range(len(memory_states)):
            memory_z.append(val)
            val *= -1     
    
    elif root.nodeType == "max" and b.winner == 0.5:
        val = 0.5
        for _ in range(len(memory_states)):
            memory_z.append(float(val))
    return b.winner, memory_states, memory_pi, memory_z


############################################# DAGGER: # TODO: bring it outside of main later
def win_loss_percentage(winnerList):
    player1 = 0
    player2 = 0
    both = 0
    for each in winnerList:
        if each ==1:
            player1 +=1
        elif each ==2:
            player2 += 1
        elif each == 0.5:
            both += 1
    return (player1, player2, both)

            

if __name__ == "__main__":
    WINNER_LIST = []
    input_shape = (4, 4, 2)
    dim_of_policy = 21
    dim_of_value = 1
    model = NN(input_shape, dim_of_policy, dim_of_value)
    model.compile_model(loss_classification='categorical_crossentropy',
                    loss_detection='mean_squared_error',
                        optimizer='adam') #should it be called once?
    max_iterations = 100
    for i in range(2):
        x1 = np.random.randint(4)
        y1 = np.random.randint(4)
        x2 = np.random.randint(4)
        y2 = np.random.randint(4)   
  
        while(x1==x2 and y1==y2):
            x1 = np.random.randint(4)
            y1 = np.random.randint(4)
            x2 = np.random.randint(4)
            y2 = np.random.randint(4) 
            # print(x1, y1, x2, y2, x3, y3)
            # WINNER_LIST.append(start(x1, y1, x2, y2, x3, y3, i))
        winner, states, pis, zs = start(1,1,3,3,1, 1, max_iterations, model)
        # print(states, pis, zs)
        dataset = []
        for each in states:
            dataset.append(create_dataset(each))

        dataset = np.array(dataset)
        pis = np.array(pis)
        zs = np.array(zs).reshape(len(zs), 1)
        print(dataset.shape)
        print(pis.shape)
        print(zs.shape)
        model.fit_model(dataset, pis, zs, epochs=50, batch_size=8)

        # AFTER ONE game => training data -> now let's train our NN
        # print(WINNER_LIST)
    print(win_loss_percentage([winner]))