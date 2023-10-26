import math
import numpy as np
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
            if unit.hitpoints !=0:
                # tmpBoard[unit.x*4 + unit.y].append(unit.name)
                self.board[unit.x*4 + unit.y] = unit.name 

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
        y1 = targetUnit.y
        if targetUnit.x == 0:
            x1 = targetUnit.x + 1
            x2 = None
            
        elif targetUnit.x == 1 or targetUnit.x == 2:
            x1 = targetUnit.x - 1
            x2 = targetUnit.x + 1
        
        elif targetUnit.x == 3:
            x1 = targetUnit.x - 1
            x2 = None    

        x3 = targetUnit.x
        if targetUnit.y ==0:
            y2 = targetUnit.y + 1
            y3 = None
            
        elif targetUnit.y ==1 or targetUnit.y==2:
            y2 = targetUnit.y - 1
            y3 = targetUnit.y + 1
        
        elif targetUnit.y ==3:
            y2 = targetUnit.y - 1
            y3 = None

        return x1, y1, x2, y1, x3, y2, x3, y3
        

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
            x1, y1, x2, y1, x3, y2, x3, y3 = self.possibleAttackPos(targetUnit) #TODO: find the shortest later / debug the passing over issue!
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
        # print(self.actionList)
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
def countUnitTypes(UnitList, q):
    unitTypes = {"ranged":0, "heavy":0, "light":0, "worker":0, "barrack":0, "base":0}
    for each in unitList:
        if isinstance(each, rangedUnit):
            unitTypes["ranged"] += 1
        elif isinstance(each, workerUnit):
            unitTypes["worker"] += 1
        elif isinstance(each, Barrack):
            unitTypes["barrack"] += 1
    return unitTypes[q]
                

class Barrack(Unit):

    def getSoldierPosition(self): #THIS IS WRONG!
        x1 = -1
        x2 = -1
        y1 = -1
        y2 = -1

        if 0<=(self.x-1) <=3:
            x1 = self.x-1
            return x1, self.y
        if 0<=(self.x+1) <=3:
            x2 = self.x+1
            return x2, self.y

        if 0<=(self.y-1) <=3:
            y1 = self.y-1
            return self.x, y1

        if 0<=(self.y+1) <=3:
            y2 = self.y+1
            return self.x, y2

    def train(self, type, ownerID, hitpoints):
        if type =="ranged":
            num = countUnitTypes(unitList, "ranged")
            x1, y1 = self.getSoldierPosition() #TODO: check if it can be placed there and it is not filled with other units
            print("LOCS for ranged", x1,y1)
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

    def canAttack(self, targetUnit):
        if self.x == targetUnit.x and abs(self.y - targetUnit.y) ==2:
            return True
        elif self.y == targetUnit.y and abs(self.x - targetUnit.x) ==2:
            return True
        else:
            return False

    def possibleAttackPos(self, targetUnit):

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
            self.actionList.append(["attack", targetUnit])

        else:
            print("NO")
            x1, y1, x2, y2 = self.possibleAttackPos(targetUnit) #TODO: find the shortest later / debug the passing over issue!
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
        # print(self.actionList)



def evaluate(action, u_me):

    if action[0] == "attack":
        u_enemy = action[1]
        # print(u_enemy.name)
        if u_me.canAttack(u_enemy):
            u_enemy.hitpoints -= 2 #TODO: remove the unit from the screen or don't let the hitpoint be less than 0;
            print("ATTACK DONE")

    elif action[0] == "buildBarrack":
        #TODO: double check if it is empty to make
        u = Barrack(action[1], action[2], "B_%s"%action[3], action[3], action[4])
        unitList.append(u)

    elif action[0] == "trainRanged":
        #TODO: double check if it is empty to make
        u = rangedUnit(action[1], action[2], "R%s"%(action[5]+2), action[3], action[4])
        unitList.append(u)
    
    else:
        action = action[0] # direction, None, None
        # print(action)
        if action == "right":
            u_me.y += 1
        elif action == "left":
            u_me.y -= 1
        elif action == "up":
            u_me.x -= 1
        elif action == "down":
            u_me.x += 1
    


def isOver(playerBasedList):
    player1 = playerBasedList[0]
    player2 = playerBasedList[1]
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

def updatePlayerBasedList(unitList):
    playerBasedList = [[],[]]
    for each in unitList:
        if each.ownerID == 1:    
            playerBasedList[0].append(each)
        elif each.ownerID == 2:    
            playerBasedList[1].append(each)
    return playerBasedList

def create_data_label(u2_pos, attackable_poses):
    
    label = np.zeros(16)
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

b = Board()
b.Draw()

unitList = [] #TODO: later, make a config and automatically add these
# u1 = rangedUnit(3, 2, "R1", 1, 6)
u2 = rangedUnit(1, 2, "R2", 2, 6)
u5 = workerUnit(3, 1, "W1", 1, 6)
# u3 = meleeUnit(0, 0, "H1", 1, 6)
# u4 = meleeUnit(1, 3, "H2", 2, 6)
# u6 = Barrack(2, 3, "BR1", 1, 6)

# unitList.append(u1)
unitList.append(u2)
unitList.append(u5)

playerBasedList = updatePlayerBasedList(unitList)
b.update_board(unitList)
b.Draw()

gameOver = False
t = 0
dataset = []

while(gameOver==False):
    for u in unitList:
        if isinstance(u, workerUnit) and countUnitTypes(unitList, "barrack")<1:
            if u.active == False: # ready to assign actions
                u.build("barrack", 3, 0, u5.ownerID, 10)
                u.active = True
        if isinstance(u, rangedUnit) or isinstance(u, meleeUnit) or isinstance(u, workerUnit): 
            if u.active == False:
                u.attackClosest(unitList)
                u.active = True
        if isinstance(u, Barrack):
            if u.active == False and countUnitTypes(unitList, "ranged")<=3:
                u.train("ranged", u.ownerID, 5)
                u.active = True

    for u in unitList:
        print(u)
        print(u.actionList)
        print("***\n")
    input()
    data = np.zeros((16, 2))
    attackable_poses = []
    for eachU in unitList:
        pos = eachU.x*4 + eachU.y
        data[pos] = [1, 0]
        if eachU.ownerID == 1:
            if u2.canAttack(eachU):
                print("CAN ATTACK "+ eachU.name)
                attackable_poses.append(pos)
    

    data, label = create_data_label(u2.x*4 + u2.y, attackable_poses)
    for eachU in unitList:
        pos = eachU.x*4 + eachU.y
        if label[pos] !=2 and eachU!=u2:
            label[pos] = 1
    print(data)
    print(label)
    dataset.append([data, label])
    input("data set")

    for eachU in unitList:
        if eachU.actionList:
            action1 = eachU.actionList.pop(0)
            print("FIRST ACTION of %s POPPED: "%eachU.name)
            if len(eachU.actionList) == 0:
                eachU.active = False
            print(action1)
            evaluate(action1, eachU)


    playerBasedList = updatePlayerBasedList(unitList)
    b.update_board(unitList)
    b.Draw()

    # check if the game is over
    if isOver(playerBasedList):
        gameOver =True

    for u in unitList:
        print(u.name, ": ", u.hitpoints)

    input("T=%s ended!"%t)

    t +=1




# Import python libraries required in this example:
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
x = np.array([[0,0,0,0,0,0,1,0,0,0,0,0,-1,0,-1,0], [1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1], [-1,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,1], [-1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,-1]])
y = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]) 

model = Sequential()
model.add(Dense(8, input_shape=(16,)))
model.add(Activation('sigmoid'))
model.add(Dense(32))
model.add(Activation('sigmoid'))
model.add(Dense(16))
model.add(Activation('softmax')) 
# Compile the model and calculate its accuracy:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
# Print a summary of the Keras model:
model.summary()


model.fit(x, y, epochs=10, batch_size=1)
# evaluate the keras model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

y_pred = model.predict([[0,0,0,1,0,0,0,0,0,0,0,0,-1,0,-1,0]])
print(y_pred)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))


# while(gameOver==False):
#     if t%10==0:
#         # u1.attackClosest(unitList)
#         # u5.build("barrack", 3, 0, u5.ownerID, 10) #TODO: check it is possible to build a barrack in that place!
#         # u2.attackClosest(unitList)
#         u6.train("ranged", u6.ownerID, 5)

#     for eachU in unitList:
#         if eachU.actionList:
#             action1 = eachU.actionList.pop(0)
#             print("FIRST ACTION of %s POPPED: "%eachU.name)
#             print(action1)
#             evaluate(action1, eachU)


#     # updating the board and printing configs
#     playerBasedList = updatePlayerBasedList(unitList)
#     b.update_board(unitList)
#     b.Draw()
#     for u in unitList:
#         print(u.name, ": ", u.hitpoints)


#     # check if the game is over
#     if isOver(playerBasedList):
#         gameOver =True

#     input("T=%s ended!"%t)

#     t +=1