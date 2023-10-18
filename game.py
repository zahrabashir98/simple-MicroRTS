from cmath import sqrt


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

# heavy, light, worker
# class meleeUnit(Unit):
#     def attack():

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
                enemyUnits.append(unit)
                distances.append(sqrt(pow((unit.x - self.x), 2) + pow((unit.y - self.y), 2)))
        targetUnitIndex = distances.index(min(distances))
        targetUnit = enemyUnits[targetUnitIndex]
        # print(enemyUnits)
        # print(distances)
        # print(targetUnit)

        if self.canAttack(targetUnit):
            print("YES")
            self.actionList.append("attack")

        else:
            print("NO")
            x1, y1, x2, y2 = self.possibleAttackPos(targetUnit) #TODO: find the shortest later / debug the passing over issue!
            # print("X1 X2\n")
            # print(x1, y1, x2, y2)
            diffX = x1 - self.x 
            diffY = y1 - self.y

            if diffX >0:
                for i in range(abs(diffX)):
                    self.actionList.append("down")
            elif diffX<0:
                for i in range(abs(diffX)):
                    self.actionList.append("up")
            
            if diffY >0:
                for i in range(abs(diffY)):
                    self.actionList.append("right")
            elif diffY<0:
                for i in range(abs(diffY)):
                    self.actionList.append("left")

            # findAttackPath()
            self.actionList.append("attack")
        print(self.actionList)



def evaluate(action, u_me, u_enemy):
    if action == "attack":
        if u_me.canAttack(u_enemy):
            u_enemy.hitpoints -= 2
            print("ATTACK DONE")
    else:
        if action == "right":
            u_me.y += 1
        elif action == "left":
            u_me.y -= 1
        elif action == "up":
            u_me.x -= 1
        elif action == "down":
            u_me.x += 1
     
def isOver(unitList):
    for each in unitList:
        if each.hitpoints <=0:
            print("GAME OVER")
            return True
    return False


###############################################################################################################
#                                                                                                             #
#                                                    Main                                                     #
#                                                                                                             #
###############################################################################################################

b = Board()
b.Draw()

unitList = []
u1 = rangedUnit(3, 3, "P1", 1, 6)
u2 = rangedUnit(1, 2, "P2", 2, 6)
unitList.append(u1)
unitList.append(u2)
b.update_board(unitList)
b.Draw()

gameOver = False
t = 0
while(gameOver==False):
    if t%10==0:
        u1.attackClosest(unitList)
        # u2.attackClosest(unitList)
    if u1.actionList:
        action1 = u1.actionList.pop(0)
        print("FIRST ACTION of U1 POPPED: "+ action1)
        evaluate(action1, u1, u2)
    if u2.actionList:
        action2 = u2.actionList.pop(0)
        print("FIRST ACTION of U2 POPPED: "+ action2)
        evaluate(action2, u2, u1)

    
    b.update_board(unitList)
    b.Draw()
    t +=1

    input("T=%s ended!"%t)
    if isOver(unitList):
        gameOver =True
