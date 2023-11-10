import math
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from random import choice
import itertools
import copy

class Board:

    def __init__(self):
        """__init_ method"""
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
        #TODO: add tie
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
        elif num_p1 == len(player1):
            self.winner = 1
        elif num_p2 == len(player2):
            self.winner = 2
        # print(self.winner)
        # input("IS OVER FUNC")
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
        
        # print("GAME IS OVER")
        return True
    
    
    def find_children(self):
        allNextStepActions = [] #CHECK LATER
        for each in self.unitList:
            if isinstance(each, rangedUnit): #TODO: support other types later
                allActions = each.allPossibleAttacks(self.unitList)
                allNextStepActions.append(set(allActions))
        # print(allNextStepActions)
          
        c = []
        for element in itertools.product(*allNextStepActions):
            c.append(element)
        all = []
        for action in c:
            all.append(self.make_move(action))
        return all

    def find_random_child(self):
        if self.is_over==True or self.isOver():
            print("IS OVER RANDOM CHILD")
            return None  # If the game is finished then no moves can be made
        # input("RANDOM CHILD\n")
        children = self.find_children()
        # empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        c = choice(children)
        # print(c.board)
        # print("THIS IS THE RANDOM CHILD")
        # c.Draw()
        # if c.unitList:
        #     print(c.unitList[0].hitpoints) #TODO: check if c has unit list
        #     print(c.unitList[1].hitpoints)
        # input("CHOSEN CHILD")
        return c

    def returnUnitbyName(self, unitList, name): #TODO: make sure we don't have duplicate names
        for u in unitList:
            if u.name == name:
                return u
        
    def make_move(self, action): # TODO: remove duplicate states? (check every attribute)
        
        # print("*********************START OF CHILDREN\n")
        tmp = copy.deepcopy(self)
        tmp.p1_actions = []
        tmp.p2_actions = []
        # print(action)
        # print("PLAYER 2")
        # print(tmp.unitList[0].x)
        # print(tmp.unitList[0].y)
        # print(tmp.unitList[0].hitpoints)
        # print("PLAYER 1")
        # print(tmp.unitList[1].x)
        # print(tmp.unitList[1].y)
        # print(tmp.unitList[1].hitpoints)
        # tmp.Draw()
        for eachU in tmp.unitList:
            if eachU.ownerID == 2:
                if action[0] == "right" and eachU.y!=3:
                    eachU.y += 1
                    tmp.p2_actions.append([action[0], None])
                elif action[0] == "left" and eachU.y!=0:
                    eachU.y -= 1
                    tmp.p2_actions.append([action[0], None])
                elif action[0] == "up" and eachU.x!=0:
                    eachU.x -= 1
                    tmp.p2_actions.append([action[0], None])
                elif action[0] == "down" and eachU.x!=3:
                    eachU.x += 1
                    tmp.p2_actions.append([action[0], None])

                elif action[0][:6] == "attack":
                    unitName = action[0][7:]
                    targetUnit = self.returnUnitbyName(tmp.unitList, unitName)
                    tmp.p2_actions.append([action[0][:6], targetUnit.name])
                    if eachU.canAttack(targetUnit):
                        targetUnit.hitpoints -= 1
        

            elif eachU.ownerID == 1:
                if action[1] == "right" and eachU.y!=3:
                    eachU.y += 1
                    tmp.p1_actions.append([action[1], None])
                elif action[1] == "left" and eachU.y!=0:
                    eachU.y -= 1
                    tmp.p1_actions.append([action[1], None])
                elif action[1] == "up" and eachU.x!=0:
                    eachU.x -= 1
                    tmp.p1_actions.append([action[1], None])
                elif action[1] == "down" and eachU.x!=3:
                    eachU.x += 1
                    tmp.p1_actions.append([action[1], None])
                elif action[1][:6] == "attack":
                    unitName = action[1][7:]
                    targetUnit = self.returnUnitbyName(tmp.unitList, unitName)
                    tmp.p1_actions.append([action[1][:6], targetUnit.name])
                    if eachU.canAttack(targetUnit):
                        targetUnit.hitpoints -= 1
                
        
        tmp.update_board()
        tmp.updatePlayerBasedList()
        # tmp.Draw()
        if tmp.unitList:
            pass
            # print(tmp.unitList)
            # print("PLAYER 2")
            # print(tmp.unitList[0].x)
            # print(tmp.unitList[0].y)
            # print(tmp.unitList[0].hitpoints)
            # print("PLAYER 1")
            # print(tmp.unitList[1].x)
            # print(tmp.unitList[1].y)
            # print(tmp.unitList[1].hitpoints)
            # print("**********************AFTER MAKING all MOVES")
        else:
            tmp.is_over = True
            # print("game is over")
        if tmp.isOver():
            tmp.is_over = True


        return tmp


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

    def canMove(self, direction):
        if direction == "l" and self.pos in [0, 4, 8, 12]:
            print("CANNOT MOVE LEFT!\n")
            return False
        elif direction == "r" and self.pos in [3, 7, 11, 15]:
            print("CANNOT MOVE RIGHT!\n")
            return False
        elif direction == "u" and self.pos in [0, 1, 2, 3]:
            print("CANNOT MOVE UP!\n")
            return False
        elif direction == "d" and self.pos in [12, 13, 14, 15]:
            print("CANNOT MOVE DOWN!\n")
            return False
        return True

    def move(self, direction):
        if self.canMove(direction):
            if direction == "l":
                self.pos -= 1
            elif direction == "r":
                self.pos += 1
            elif direction == "u":
                self.pos -= 4
            elif direction == "d":
                self.pos += 4

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
            # self.actionList.append(["trainRanged2", x1, y1, ownerID, hitpoints, num])
            # self.actionList.append(["trainRanged3", x1, y1, ownerID, hitpoints, num])

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

class workerUnit(meleeUnit):
    def collect():
        pass

    def build(self, type, x, y, ownerID, hitpoints):
        if type == "barrack": #TODO: add exceptions for not being able to add a unit at a specific position becuase it is full!
            self.actionList.append(["buildBarrack", x, y, ownerID, hitpoints])
            # self.actionList.append(["buildBarrack2", x, y, ownerID, hitpoints])
            # self.actionList.append(["buildBarrack3", x, y, ownerID, hitpoints])
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

        if len(unitList) >1:
            targetUnitIndex = distances.index(min(distances))
            targetUnit = enemyUnits[targetUnitIndex]


            if self.canAttack(targetUnit):
                # print("YES")
                self.actionList.append(["attack", targetUnit]) # the last to be droppped and applied
                # self.actionList.append(["attack2", targetUnit])

            else:
                # print("NO")
                x1, y1 = self.possibleAttackPos(targetUnit) #TODO: find the shortest later / debug the passing over issue!
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

                self.actionList.append(["attack", targetUnit])
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

###############################################################################################################
#                                                                                                             #
#                                                    Main                                                     #
#                                                                                                             #
###############################################################################################################
# assumption: I'm assuming all actions' durations are equal to one time step!
from monte_carlo_tree_search import MCTS, Node
def start(x1, y1, x2, y2, flag):
    b = Board()
    b.Draw()

    u2 = rangedUnit(x1, y1, "R2", 2, 5)
    u5 = rangedUnit(x2, y2, "R1", 1, 5)
    b.unitList.append(u2)
    b.unitList.append(u5)
    print(b.unitList)

    b.updatePlayerBasedList()
    print(b.playerBasedList)
    b.update_board()
    b.Draw()

    ##############################
    tree = MCTS()

    # for j in range(10):
    #     tree.do_rollout(b) 
    # bestAction = tree.choose(b)
    # print(bestAction)
    # if bestAction[0][1]:
    #     print((bestAction[0][1].name))
    # exit()
    gameOver = False
    t = 0
    dataset = []
    labels = []
    count = 0
    while(gameOver==False):
        for u in b.unitList:
            if u ==u2: # the player we are intertsed to learn its behaviour! (Ranged unit for now, and the behaviour is "attack closeset" for now!)
                for j in range(100):
                    tree.do_rollout(b) 
                bestAction = tree.choose(b)
                print(bestAction)
                # print(tree.Q)
                input("LOOK AT THIS")
                if u.active == False:
                    u.actionList.append(bestAction[0])
                    u.active = True

            else:
                # u.attackClosest(b.unitList)
                if isinstance(u, workerUnit) and countUnitTypes(b.unitList, "barrack")<1:
                    if u.active == False: # ready to assign actions
                        x1 = np.random.randint(4) 
                        y1 = np.random.randint(4)
                        while(b.board[4*x1 + y1]!="  "):
                            x1 = np.random.randint(4)  
                            y1 = np.random.randint(4)
                        u.build("barrack", x1, y1, u5.ownerID, 10)
                        u.active = True

                if t%5 == 0 and isinstance(u, rangedUnit) or isinstance(u, meleeUnit) or isinstance(u, workerUnit): 
                    if u.active == False:
                        u.attackClosest(b.unitList)
                        u.active = True
                if t%5 != 0 and isinstance(u, rangedUnit) or isinstance(u, meleeUnit) or isinstance(u, workerUnit): 
                    if u.active == False:
                        u.actionList.append([np.random.choice(["left", "right", "up", "down"]), None])
                        u.active = True
                if isinstance(u, Barrack):
                    if u.active == False and countUnitTypes(b.unitList, "ranged")<=3:
                        u.train("ranged", u.ownerID, 5, b.unitList, b.board)
                        u.active = True

        # for u in unitList:
        #     print(u)
        #     print(u.actionList)
        #     print("***\n")
        # input()

        data = np.zeros((4,4, 2))
        label= np.zeros(21)
        flag = False

        for eachU in b.unitList:
            if eachU == u2:
                data[u2.x][u2.y] = [1, 0]
                action = eachU.actionList[0]
                if action[0] == "attack":
                    if isinstance(action[1], Unit):
                        u_enemy = action[1]
                    else:
                        u_enemy = b.returnUnitbyName(b.unitList, action[1])
                    label[(u_enemy.x*4) + u_enemy.y] = 1
                    flag = True
                elif action[0] == "left":
                    label[16] = 1
                    flag = True
                elif action[0] == "right":
                    label[17] = 1
                    flag = True
                elif action[0] == "up":
                    label[18] = 1
                    flag = True
                elif action[0] == "down":
                    label[19] = 1
                    flag = True
            else:
                data[eachU.x][eachU.y] = [0, 1]
        
        if flag == False:    
            label[20] = 1 #do nothing

        # print(data)
        # print(label)
        print("\nLET'S EVALUATE\n\n")
        #evaluate
        for eachU in b.unitList:
            if eachU.actionList:
                action1 = eachU.actionList.pop(0)
                print("FIRST ACTION of %s POPPED: "%eachU.name)
                if len(eachU.actionList) == 0:
                    eachU.active = False
                print(action1)
                evaluate(action1, eachU, b, b.board)

        print(np.array(data).shape)


        dataset.append(data)
        labels.append(label)

        b.updatePlayerBasedList()

        b.update_board()
        b.Draw()
        print("BOARD E ASLI bood\n")
        # b.find_children()
        # input("NEXT FIND CHILDREN for the next play")
 
        if b.isOver():
            gameOver =True
            b.is_over = True

        for u in b.unitList:
            print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

        # print(t)
        # input("T=%s ended!"%t)
        t +=1

    # print(len(dataset))
    # print(np.array(dataset).shape)
    # print(len(labels))
    return dataset, labels

############################################# DAGGER: # TODO: bring it outside of main later


if __name__ == "__main__":
    x1 = np.random.randint(4)
    y1 = np.random.randint(4)
    x2 = np.random.randint(4)
    y2 = np.random.randint(4)   
    while(x1==x2 and y1==y2):
        x1 = np.random.randint(4)
        y1 = np.random.randint(4)
        x2 = np.random.randint(4)
        y2 = np.random.randint(4)   

    print(x1, y1, x2, y2)
    dataset, labels = start(x1, y1, x2, y2, 0)