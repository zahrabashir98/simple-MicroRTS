import math
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization

class Board:

    def __init__(self):
        """__init_ method"""
        self.x = "-"
        self.y = "|"
        self.z = "  "
        self.zz = ""

        self.board = ["  "]*16
        self.number_of_units = 2

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

    def update_board(self, unitList):
        self.board = ["  "]*16
        tmpBoard = {key: [] for key in range(16)}
        # print(tmpBoard)
        for unit in unitList:
            if unit.hitpoints >0:
                # tmpBoard[unit.x*4 + unit.y].append(unit.name)
                self.board[(unit.x*4) + unit.y] = unit.name 
            elif unit.hitpoints <=0:
                unitList.remove(unit)
        return unitList
    

        
        # for u in tmpBoard:
        #     name  = ""
        #     for each in tmpBoard[u]:
        #         name += each
        #     self.board[u] = name   # unit[0] = x


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

        # if self.x == targetUnit.x and abs(self.y - targetUnit.y) ==2:
        #     return True
        # elif self.y == targetUnit.y and abs(self.x - targetUnit.x) ==2:
        #     return True
        # else:
        #     return False

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
        

    def simulateAttackClosest(self, unitList): #TODO: never tells you to do nothing?
        enemyUnits = []
        distances = []
        for unit in unitList:
            if unit.ownerID != self.ownerID:
                # print(unit.name)
                enemyUnits.append(unit)
                distances.append(math.sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        # print(distances)
        targetUnitIndex = distances.index(min(distances))
        targetUnit = enemyUnits[targetUnitIndex]

        if self.canAttack(targetUnit):
            print("YES")
            targetPos = targetUnit.x*4 + targetUnit.y
            return targetPos

        else:
            print("NO")
            x1, y1= self.possibleAttackPos(targetUnit) #TODO: find the shortest later / debug the passing over issue!
            diffX = x1 - self.x 
            diffY = y1 - self.y

            if diffX >0:
                return 19#"down"
            elif diffX<0:
                return 18 #"up"
            if diffY >0:
                return 17 #"right"
            elif diffY<0:
                return 16 #"left"

    def attackClosest(self, unitList):
        enemyUnits = []
        distances = []
        for unit in unitList:
            if unit.ownerID != self.ownerID:
                # print(unit.name)
                enemyUnits.append(unit)
                distances.append(math.sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        # print(distances)
        targetUnitIndex = distances.index(min(distances))
        targetUnit = enemyUnits[targetUnitIndex]
        # print(enemyUnits)
        # print(distances)
        # print(targetUnit)

        if self.canAttack(targetUnit):
            print("YES")
            self.actionList.append(["attack", targetUnit]) # the last to be droppped and applied
            # self.actionList.append(["attack2", targetUnit])

        else:
            print("NO")
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



def evaluate(action, u_me, unitList, board):

    if action[0] == "attack":
        u_enemy = action[1]
        # print(u_enemy.name)
        if u_me.canAttack(u_enemy):
            u_enemy.hitpoints -= 1 #TODO: remove the unit from the screen or don't let the hitpoint be less than 0; #DONE
            print("ATTACK to %s DONE"%u_enemy.name)
        else:
            print("attack not done")

    elif action[0] == "buildBarrack":
        #TODO: double check if it is empty to make; #DONE
        u = Barrack(action[1], action[2], "B%s"%action[3], action[3], action[4])
        unitList.append(u)


    elif action[0] == "trainRanged":
        #TODO: double check if it is empty to make; #DONE
        u = rangedUnit(action[1], action[2], "R%s"%(action[5]+2), action[3], action[4])
        unitList.append(u)  

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
    


def isOver(playerBasedList, unitList):
    player1 = playerBasedList[0]
    player2 = playerBasedList[1]
    if len(unitList) <=1:
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
    print("GAME IS OVER")
    return True

def updatePlayerBasedList(unitList):
    playerBasedList = [[],[]]
    for each in unitList:
        if each.ownerID == 1:    
            playerBasedList[0].append(each)
        elif each.ownerID == 2:    
            playerBasedList[1].append(each)
    return playerBasedList

def create_data_label(data, u2_pos, attackable_poses):
    
    label = np.zeros(17)
    data[u2_pos] = [0, 1]
    for pos in attackable_poses:
        label[pos] = 2 # attackable units
    return data, label
###############################################################################################################
#                                                                                                             #
#                                                    Main                                                     #
#                                                                                                             #
###############################################################################################################
# assumption: I'm assuming all actions' durations are equal to one time step!
def start(x1, y1, x2, y2, flag):
    b = Board()
    b.Draw()

    unitList = [] #TODO: later, make a config and automatically add these
    if flag == 0:
        u2 = rangedUnit(x1, y1, "R2", 2, 20)
        u5 = workerUnit(x2, y2, "W1", 1, 20)
        # u6 = workerUnit(0, 3, "W1", 1, 6)
        # u3 = meleeUnit(0, 0, "H1", 1, 6)
        # u4 = meleeUnit(1, 3, "H2", 2, 6)
        # u6 = Barrack(2, 3, "BR1", 1, 6)
        unitList.append(u2)
        unitList.append(u5)

    elif flag == 1:
        if x1 == x2 and y1 ==y2:
            assert 1!=1
        u2 = rangedUnit(x1, y1, "R2", 2, 15)
        u6 = Barrack(x2, y2, "BR1", 1, 10)
        unitList.append(u2)
        unitList.append(u6)
    playerBasedList = updatePlayerBasedList(unitList)
    unitList = b.update_board(unitList)
    b.Draw()

    gameOver = False
    t = 0
    dataset = []
    labels = []
    count = 0
    while(gameOver==False):
        for u in unitList:
            if u ==u2: # the player we are intertsed to learn its behaviour! (Ranged unit for now, and the behaviour is "attack closeset" for now!)
                if u.active == False:
                    u.attackClosest(unitList)
                    u.active = True
            else:
                if isinstance(u, workerUnit) and countUnitTypes(unitList, "barrack")<1:
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
                        u.attackClosest(unitList)
                        u.active = True
                if t%5 != 0 and isinstance(u, rangedUnit) or isinstance(u, meleeUnit) or isinstance(u, workerUnit): 
                    if u.active == False:
                        u.actionList.append([np.random.choice(["left", "right", "up", "down"]), None])
                        u.active = True
                if isinstance(u, Barrack):
                    if u.active == False and countUnitTypes(unitList, "ranged")<=3:
                        u.train("ranged", u.ownerID, 5, unitList, b.board)
                        u.active = True

        # for u in unitList:
        #     print(u)
        #     print(u.actionList)
        #     print("***\n")
        # input()

        data = np.zeros((4,4, 2))
        label= np.zeros(21)
        flag = False

        for eachU in unitList:
            if eachU == u2:
                data[u2.x][u2.y] = [1, 0]
                action = eachU.actionList[0]
                if action[0] == "attack":
                    u_enemy = action[1]
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
        for eachU in unitList:
            if eachU.actionList:
                action1 = eachU.actionList.pop(0)
                print("FIRST ACTION of %s POPPED: "%eachU.name)
                if len(eachU.actionList) == 0:
                    eachU.active = False
                print(action1)
                evaluate(action1, eachU, unitList, b.board)

        print(np.array(data).shape)


        dataset.append(data)
        labels.append(label)
        # if t==0:
        #     dataset.append(data)
        #     labels.append(label)
        # else:
        #     for each in dataset:
        #         if data.all() == each.all():
        #             print("NOT ADDED")
        #             # input()
        #         else:
        #             dataset.append(data)
        #             labels.append(label)
        playerBasedList = updatePlayerBasedList(unitList)
        unitList = b.update_board(unitList)
        b.Draw()
        # if flag ==1:
        #     input()

        # check if the game is over
        if isOver(playerBasedList, unitList):
            gameOver =True

        for u in unitList:
            print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

        # print(t)
        # input("T=%s ended!"%t)
        t +=1


    # print(len(dataset))
    # print(np.array(dataset).shape)
    # print(len(labels))
    return dataset, labels


def simulate(x1, y1, x2, y2, model):
    b = Board()
    b.Draw()
    unitList = []
    u2 = rangedUnit(x1, y1, "R2", 2, 20)
    u5 = workerUnit(x2, y2, "W1", 1, 10)
    unitList.append(u2)
    unitList.append(u5)
    playerBasedList = updatePlayerBasedList(unitList)
    unitList = b.update_board(unitList)
    b.Draw()

    gameOver = False
    t = 0
    dataset = []
    labels = []

    while(gameOver==False):
        data = np.zeros((4,4, 2))
        label= np.zeros(21)

        for eachU in unitList:
            if eachU == u2:
                data[u2.x][u2.y] = [1, 0]
            else:
                data[eachU.x][eachU.y] = [0, 1]
        
        for u in unitList:
            if u ==u2: # the player we are intertsed to learn its behaviour! (Ranged unit for now, and the behaviour is "attack closeset" for now!)
                print("\n\n***************\n")
                if u.active == False:
                    output = model.predict(data.reshape(1,4,4,2))
                    result = np.random.choice(np.arange(0, 21), p=output[0])
                    print(result)
                    if result>=0 and result<=15:
                        targetUnit = u5
                        for every in unitList:
                            if (every.x*4 + every.y) == result:
                                targetUnit = every
                        u.actionList.append(["attack", targetUnit])
                        # u.actionList.append(["attack2", targetUnit])
                    elif result == 16:
                        u.actionList.append(["left", None])
                    elif result == 17:
                        u.actionList.append(["right", None])
                    elif result == 18:
                        u.actionList.append(["up", None])
                    elif result == 19:
                        u.actionList.append(["down", None])
                    elif result == 20:
                        u.actionList.append(["nothing", None])
                    
                    labelIndex = u.simulateAttackClosest(unitList)
                    print(labelIndex)
                    label[labelIndex] = 1
                    print(label)
                    print(u.actionList)
                    # input("IN THE U2 prediction loop")
                    u.active = True
            else:
                if isinstance(u, workerUnit) and countUnitTypes(unitList, "barrack")<1:
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
                        u.attackClosest(unitList)
                        u.active = True
                if t%5 != 0 and isinstance(u, rangedUnit) or isinstance(u, meleeUnit) or isinstance(u, workerUnit): 
                    if u.active == False:
                        u.actionList.append([np.random.choice(["left", "right", "up", "down"]), None])
                        u.active = True
                if isinstance(u, Barrack):
                    if u.active == False and countUnitTypes(unitList, "ranged")<=3:
                        u.train("ranged", u.ownerID, 5, unitList, b.board)
                        u.active = True


        #evaluate
        for eachU in unitList:
            if eachU.actionList:
                action1 = eachU.actionList.pop(0)
                print("FIRST ACTION of %s POPPED: "%eachU.name)
                if len(eachU.actionList) == 0:
                    eachU.active = False
                print(action1)
                evaluate(action1, eachU, unitList, b.board)


        dataset.append(data)
        labels.append(label)

        playerBasedList = updatePlayerBasedList(unitList)
        unitList = b.update_board(unitList)
        b.Draw()

        # check if the game is over
        if isOver(playerBasedList, unitList):
            gameOver =True

        for u in unitList:
            print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

        # print(t)
        # input("T=%s ended!"%t)
        t +=1


    print(len(dataset))
    print(np.array(dataset).shape)
    print(len(labels))
    return np.array(dataset), np.array(labels)


############################################# DAGGER: # TODO: bring it outside of main later
def main():
    trainingData = []
    trainingLabels = []
    validData = []
    validLabels = []
    testData = []
    testLabels = []
    allData = []
    allLabels = []
    num = 200
    import numpy as np

    for k in range(num):
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
        # input()
        if k <=100:
            tmpd, tmpl = start(x1, y1, x2, y2, 0)
        elif k >100:
            tmpd, tmpl = start(x1, y1, x2, y2, 1)
        # alert = False
        # if len(allData) != 0:
        #     for new in tmpd:
        #         alert = False
        #         # print(allData.shape)
        #         # print(np.array(tmpd).shape)
        #         # input()
        #         for old in allData:
        #             if np.array_equal(old, new):
        #                 alert = True
        #         if alert == False:
        #             allData = np.concatenate((allData, tmpd))
        #             allLabels += tmpl
        # else:
        #     allData += tmpd
        #     allLabels += tmpl
        #     allData = np.array(allData)
        # print(allData.shape)
        # input("ALLDATA")

        allData += tmpd
        allLabels += tmpl

        # dataset, labels = start(x1, y1, x2, y2)
        # if i<num/3:
        #     trainingData+= dataset
        #     trainingLabels += labels
        # elif i>=num/3 and i<2*num/3:
        #     validData += dataset
        #     validLabels += labels
        # elif i>=2*num/3:
        #     testData += dataset
        #     testLabels += labels

#TODO: action durations
############################################# (NN)

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
    import numpy as np
    from matplotlib import pyplot as plt
    import keras
    # allData = allData.tolist()
    c = list(zip(allData, allLabels))
    np.random.shuffle(c)
    allData, allLabels = zip(*c)

    x = np.array(allData[0:8*int(len(allData)/10)])
    y = np.array(allLabels[0:8*int(len(allLabels)/10)])

    valData = np.array(allData[8*int(len(allData)/10): 9*int(len(allData)/10)])
    valLabels = np.array(allLabels[8*int(len(allLabels)/10): 9*int(len(allLabels)/10)])

    testData = np.array(allData[8*int(len(allData)/10): 9*int(len(allData)/10)])
    testLabels = np.array(allLabels[8*int(len(allLabels)/10): 9*int(len(allLabels)/10)])
    # x = np.array(trainingData)
    # y = np.array(trainingLabels)
    # valData = np.array(validData)
    # valLabels = np.array(validLabels)
    # testData = np.array(testData)
    # testLabels = np.array(testLabels)
    print(x.shape)
    print(y.shape)
    print(valData.shape)
    print(valLabels.shape)
    print(testData.shape)
    print(testLabels.shape)
    input()

    num_filters = 32
    filter_size = 1
    pool_size = 1

    # add more layers?
    model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(4, 4, 2)),
    BatchNormalization(),
    Dropout(0.25),
    MaxPooling2D(pool_size=pool_size),
    Conv2D(32, 1),
    Dropout(0.3),
    Flatten(),
    Dense(21, activation='softmax'),
    ])

    model.compile(
    keras.optimizers.Adam(lr=0.1),
    #   'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],

    )

    history = model.fit(
    x,
    y,
    epochs=200,
    validation_data = (valData, valLabels),
    batch_size = 128
    # batch_size=2
    )
    input()

    dagger = False
    if dagger:
        input("BEFORE FITTING")
        for i in range(2):
            input("ROUND%s:  of training"%str(i+1))
            for j in range(30):
                x1 = np.random.randint(4)
                y1 = np.random.randint(4)
                x2 = np.random.randint(4)
                y2 = np.random.randint(4)   
                while(x1==x2 and y1==y2):
                    x1 = np.random.randint(4)
                    y1 = np.random.randint(4)
                    x2 = np.random.randint(4)
                    y2 = np.random.randint(4)   

                if j%9==0:
                    newValidDataset, newValidLabels = simulate(x1, y1, x2, y2, model)
                    print(newValidDataset.shape)
                    print(newValidLabels.shape)
                elif j%8==0:
                    newTestDataset, newTestLabels = simulate(x1, y1, x2, y2, model)
                    print(newTestDataset.shape)
                    print(newTestLabels.shape)
                else:
                    newTrainingDataset, newTrainingLabels = simulate(x1, y1, x2, y2, model) #TODO: make the opponent more different
                    print(newTrainingDataset.shape)
                    print(newTrainingLabels.shape)
            

            print(newTrainingDataset.shape)
            print(newTrainingLabels.shape)
            print(newValidDataset.shape)
            print(newValidLabels.shape)
            print(newTestDataset.shape)
            print(newTestLabels.shape)
            x = np.concatenate((x, newTrainingDataset))
            y = np.concatenate((y, newTrainingLabels))
            valData = np.concatenate((valData, newValidDataset))
            valLabels = np.concatenate((valLabels, newValidLabels))
            testData = np.concatenate((testData, newTestDataset))
            testLabels = np.concatenate((testLabels, newTestLabels))   
            print(x.shape)
            print(y.shape)
            input()
            history = model.fit(
                x,
                y,
                epochs=100,
                validation_data = (valData, valLabels),
                shuffle = True)

        # print(history.history['val_accuracy'])
        input()

    import matplotlib.pyplot as plt

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


    test_loss, test_acc = model.evaluate(testData,  testLabels, verbose=2)
    print(test_loss)
    print(test_acc)
    # test = np.zeros((1, 4,4,2))
    # test[0][0][0][0] = 1
    # test[0][2][2][1] = 1
    # test[0][0][3][1] = 1
    # print(test)


    # label = model.predict(x)
    # print(label)
    # result = np.random.choice(np.arange(0, 21), p=label[0])
    # print(result)


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold as SKF
    from sklearn import metrics

    testData = np.array(valData)
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    testData = testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2]*testData.shape[3])
    wholeData = np.concatenate((x, testData))
    print(wholeData.shape)
    wholeLabels = []
    for i in range(len(x)):
        wholeLabels.append(1)
    for i in range(len(testData)):
        wholeLabels.append(0)

    wholeLabels = np.array(wholeLabels)
    print(wholeLabels.shape)

    m = RandomForestClassifier(n_jobs=-1, max_depth=5, min_samples_leaf = 5)
    predictions = np.zeros(wholeLabels.shape) #creating an empty prediction array

    x = wholeData
    y = wholeLabels

    skf = SKF(n_splits=20, shuffle=True, random_state=100)
    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_test)[:, 1] #calculating the probability
        predictions[test_idx] = probs


    print("ROC-AUC for train and test distributions:", metrics.auc(y, predictions)) 
    # if it is not very high(>0.8), it is ok, and means that we don't have covariate shift in our dataset
    # based on: https://towardsdatascience.com/how-dis-similar-are-my-train-and-test-data-56af3923de9b

if __name__ == "__main__":
    main()