import math

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
        for unit in unitList:
            self.board[unit.x*4 + unit.y] = unit.name # unit[0] = x


class Unit:
    def __init__(self, x, y, name, ownerID, hitpoints):
        """__init_ method"""
        self.x = x
        self.y = y
        self.name = name
        self.ownerID = ownerID
        self.hitpoints = hitpoints
        self.actionList = []

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
        print(distances)
        print(targetUnit.name)

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
        print(self.actionList)

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
class workerUnit(meleeUnit):
    def collect():
        pass
    def build():
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
                print(unit.name)
                enemyUnits.append(unit)
                distances.append(math.sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        print(distances)
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
        print(self.actionList)



def evaluate(action, u_me):

    if action[0] == "attack":
        u_enemy = action[1]
        print(u_enemy.name)
        if u_me.canAttack(u_enemy):
            u_enemy.hitpoints -= 2
            print("ATTACK DONE")
    else:

        action = action[0] # direction, None, None
        print(action)
        print("INJA")
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


###############################################################################################################
#                                                                                                             #
#                                                    Main                                                     #
#                                                                                                             #
###############################################################################################################
# assumption: I'm assuming all actions durations are equal to one time step!

b = Board()
b.Draw()

unitList = [] #TODO: later, make a config and automatically add these
u1 = rangedUnit(3, 3, "R1", 1, 6)
u2 = rangedUnit(1, 2, "R2", 2, 6)
u3 = meleeUnit(0, 0, "H1", 1, 6)
u4 = meleeUnit(1, 3, "H2", 2, 6)

unitList.append(u1)
unitList.append(u2)
unitList.append(u3)
unitList.append(u4)

playerBasedList = [[],[]]
for each in unitList:
    if each.ownerID == 1:    
        playerBasedList[0].append(each)
    elif each.ownerID == 2:    
        playerBasedList[1].append(each)
# print(playerBasedList)
b.update_board(unitList)
b.Draw()

gameOver = False
t = 0
while(gameOver==False):
    if t%10==0:
        u4.attackClosest(unitList)
        # u2.attackClosest(unitList)

    for eachU in unitList:
        if eachU.actionList:
            action1 = eachU.actionList.pop(0)
            print("FIRST ACTION of %s POPPED: "%eachU.name)
            print(action1)
            evaluate(action1, eachU)


    # updating the board and printing configs
    b.update_board(unitList)
    b.Draw()
    for u in unitList:
        print(u.name, ": ", u.hitpoints)


    # check if the game is over
    if isOver(playerBasedList):
        gameOver =True
    input("T=%s ended!"%t)
    t +=1