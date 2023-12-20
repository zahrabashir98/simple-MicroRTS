import numpy as np
from random import choice
import itertools
import copy
import random
from collections import namedtuple
from MCTS import MCTS, Node
from NN import *
from tensorflow.keras.models import model_from_json
from Unit import *

"""
Note: in thie implementation, the first one who plays is player2 in the max node :D
"""
# np.random.seed(1)
# random.seed(1)
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
    
    
    def find_children(self):
        if self.isOver():  # If the game is finished then no moves can be made
            return [[],[], []] #set()
        # assert self.nodeType == player
        allActions = []
        if self.nodeType == "max":
            for each in self.unitList:
                if isinstance(each, rangedUnit): #TODO: support other types later
                    if each.ownerID == 2:
                        allActions = set(each.allPossibleActions(self.unitList))
        elif self.nodeType == "min":
            for each in self.unitList:
                if isinstance(each, rangedUnit): #TODO: support other types later
                    if each.ownerID == 1:
                        allActions = set(each.allPossibleActions(self.unitList))
                        # allActions = set(each.allPossibleAttacks(self.unitList)) # this is the case it expands the children based on attack_closest only

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
            return None  # If the game is finished then no moves can be made

        childrenObjs, _, _ = self.find_children()
        c = choice(childrenObjs)
        return c

    def returnUnitbyName(self, unitList, name): #TODO: make sure we don't have duplicate names
        for u in unitList:
            if u.name == name:
                return u
    
    def returnUnitbyPos(self, unitList, x , y): #TODO: make sure we don't have duplicate names
        for u in unitList:
            if u.x == x and u.y==y:
                return u
    
    def check_if_it_can_move(self, pos):
        if self.board[pos] == "  ":
            return True
        else:
            return False
    

    def make_move(self, action): # TODO: remove duplicate states? (check every attribute)
        #TODO: this is for only 2 unit games. we have to extend it to other for than two unit ones later.
        

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
                        tmp.p2_actions.append([action[:6], unitName])
                        index = targetUnit.x * 4 + targetUnit.y
                    else:
                        index = 20
                        assert 8!=8

        elif self.nodeType == "min":
            tmp = copy.deepcopy(self)
            tmp.nodeType = "max"
            tmp.p1_actions = []
            tmp.p2_actions = self.p2_actions #get it from your father

            for eachU in tmp.unitList: #TODO: if changed to multi-unit -> this should be changed and add to queue for executing actions (not applying immediately)
                if eachU.ownerID == 2:
                    if self.p2_actions[0][0] == "right" and eachU.y!=3:
                        if self.check_if_it_can_move(eachU.x*4 + eachU.y+1):
                            eachU.y += 1
                    elif self.p2_actions[0][0]== "left" and eachU.y!=0:
                        if self.check_if_it_can_move(eachU.x*4 + eachU.y-1):
                            eachU.y -= 1
                    elif self.p2_actions[0][0] == "up" and eachU.x!=0:
                        if self.check_if_it_can_move((eachU.x-1)*4 + eachU.y):
                            eachU.x -= 1
                    elif self.p2_actions[0][0] == "down" and eachU.x!=3:
                        if self.check_if_it_can_move((eachU.x+1)*4 + eachU.y):
                            eachU.x += 1

                    elif self.p2_actions[0][0] == "attack":
                        unitName = self.p2_actions[0][1]
                        targetUnit = tmp.returnUnitbyName(tmp.unitList, unitName)
                        targetPrevUnit = self.returnUnitbyName(self.unitList, unitName)
                        if eachU.canAttack(targetPrevUnit):
                            targetUnit.hitpoints -= 1
                        
            
                elif eachU.ownerID == 1:
                    if action == "right" and eachU.y!=3:
                        tmp.p1_actions.append([action, None])
                        index = 17
                        if self.check_if_it_can_move(eachU.x*4 + eachU.y+1):
                            eachU.y += 1
                            
                            
                    elif action == "left" and eachU.y!=0:
                        tmp.p1_actions.append([action, None])
                        index = 16
                        if self.check_if_it_can_move(eachU.x*4 + eachU.y-1):
                            eachU.y -= 1
                            
                            
                    elif action == "up" and eachU.x!=0:
                        tmp.p1_actions.append([action, None])
                        index = 18
                        if self.check_if_it_can_move((eachU.x-1)*4 + eachU.y):
                            eachU.x -= 1
                            
                            
                    elif action == "down" and eachU.x!=3:
                        tmp.p1_actions.append([action, None])
                        index = 19
                        if self.check_if_it_can_move((eachU.x+1)*4 + eachU.y):
                            eachU.x += 1
                            

                    elif action[:6] == "attack":
                        unitName = action[7:]
                        targetUnit = tmp.returnUnitbyName(tmp.unitList, unitName)
                        targetPrevUnit = self.returnUnitbyName(self.unitList, unitName)
                        tmp.p1_actions.append([action[:6], unitName])
                        if eachU.canAttack(targetPrevUnit):
                            targetUnit.hitpoints -= 1
                        index = targetUnit.x * 4 + targetUnit.y
                    
                    else:
                        index = 20
                        assert 7!=7
                              
            tmp.update_board()
            tmp.updatePlayerBasedList()
        
        if tmp.unitList:
            pass

        else:
            tmp.is_over = True

        if tmp.isOver():
            tmp.is_over = True

        return tmp, tmp.hash(), index



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
    data = np.zeros((2, 4, 4))
    fourth_layer = np.zeros((1, 4, 4))
    if board.nodeType == "max":
        for eachU in board.unitList:
            if eachU.ownerID == 2:
                data[0][eachU.x][eachU.y] = 1
                # fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints
            else:
                data[1][eachU.x][eachU.y] = 1
                # fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints

        third_layer = np.zeros((1,4,4)) #Player turn to play - o for p2 (first who plays:D)
        # data = np.concatenate((data,third_layer, fourth_layer))
        data = np.concatenate((data,third_layer))


    elif board.nodeType == "min":
        for eachU in board.unitList:
            if eachU.ownerID == 2:
                data[0][eachU.x][eachU.y] = 1
                # fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints
            else:
                data[1][eachU.x][eachU.y] = 1
                # fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints

        third_layer = np.ones((1,4,4))
        # data = np.concatenate((data,third_layer, fourth_layer))
        data = np.concatenate((data,third_layer))

    return data
###############################################################################################################
#                                                                                                             #
#                                                    Main                                                     #
#                                                                                                             #
###############################################################################################################
# assumption: I'm assuming all actions' durations are equal to one time step!

def start(x1, y1, x2, y2, max_iterations, model, itr):
    b = Board()

    # h1 = np.random.randint(1,6) #1-5
    # h2 = np.random.randint(1,6) #1-5
    # print(h1, h2,x1, y1,x2, y2)
    # u2 = rangedUnit(0, 0, "R2", 2, 3)
    # u5 = rangedUnit(0, 2, "R1", 1, 3)
    x1 = 1
    y1 = 1
    x2 = 3
    y2 = 3
    h1 = 3
    h2 = 3
    u2 = rangedUnit(x1, y1, "R2", 2, h1)
    u5 = rangedUnit(x2, y2, "R1", 1, h2)
    b.unitList.append(u2)
    b.unitList.append(u5)
    b.updatePlayerBasedList()
    b.update_board()
    b.Draw()
    tree = MCTS()
    gameOver = False
    t = 0
    maxFlag = False

    memory_states  = []
    memory_pi = []
    memory_z = []
    root = b
    tree.Pi[root] = 1 #assigning root to 1

    if itr ==0:
        VanillaMCTS = False
    else:
        VanillaMCTS= False

    print("++++++++++++++++++ Begining of one game +++++++++++++++\n\n")
    while(gameOver==False):
        print("T = ", t)
        for u in b.unitList:

            if u.ownerID==2: # we should call first max and then min node
                print("PLAYER MAX")
                memory_states.append(b)

                for i in range(10000):
                    tree.do_rollout(b, model, VanillaMCTS) 

                pi = tree.returnDist(b)
                print("distribution from game tree search:")
                print(pi)
                memory_pi.append(pi)
                b = tree.choose_based_on_dist(b, pi) #ALTERNATIVE OPTION IS TO CALL choose(b) -> it chooses the best child based on Val/N
                print(b.p2_actions)
                input("enter 'return' for the next iteration\n")


            else:
                print("PLAYER MIN")
                memory_states.append(b)
                for j in range(10000):
                    tree.do_rollout(b, model, VanillaMCTS)

                pi = tree.returnDist(b)
                print("distribution from game tree search:")
                print(pi)
                memory_pi.append(pi)
                b = tree.choose_based_on_dist(b, pi)
                print(b.p1_actions)
                input("enter 'return' for the next iteration\n")

        b.updatePlayerBasedList()
        b.update_board()
        b.Draw()
 
        if b.isOver():
            gameOver =True
            b.is_over = True
        for u in b.unitList:
            print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

        t +=1
        print("******\n")
        if t>=max_iterations:
            print("MAX IETR!")
            #TODO: check how often?
            b.is_over = True
            b.winner = 0.5
            maxFlag = True
            break
 
    # check winning status at the end to collect the z values. Tha game always starts with a max root.
    # (The win/loss is always happening in the max node too. In the min node we don't apply any changes.)
    if root.nodeType == "max" and b.winner == 2:
        val = 1
        for _ in range(len(memory_states)):
            memory_z.append(val)
            # val *= -1

    elif root.nodeType == "max" and b.winner == 1:
        val = -1
        print("LOST!")
        for _ in range(len(memory_states)):
            memory_z.append(val)
            # val *= -1     
    
    elif root.nodeType == "max" and b.winner == 0.5:
        val = 0.5
        for _ in range(len(memory_states)):
            memory_z.append(val)
            val *= -1

    return b.winner, memory_states, memory_pi, memory_z, maxFlag



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

    # input_shape = (4, 4, 4) #for the case we had hitpoints
    input_shape = (3, 4, 4)
    dim_of_policy = 21
    dim_of_value = 1
    model = NN(input_shape, dim_of_policy, dim_of_value)
    model.compile_model(loss_classification='categorical_crossentropy',
                    loss_detection='mean_squared_error',
                        optimizer='adam' #k.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)  #, clipvalue=0.5
                        )
    max_iterations = 40
    t = [i for i in range(15,-1, -1)]
    combinations = list(itertools.combinations(t, 2))
    
    training = True
    if training:
        for itr in range(5):
            dataset = []
            pi_list = []
            z_list = []
            for i in range(1):
                x1 = np.random.randint(4)
                y1 = np.random.randint(4)
                x2 = np.random.randint(4)
                y2 = np.random.randint(4)
                while(x1==x2 and y1==y2):
                    x1 = np.random.randint(4)
                    y1 = np.random.randint(4)
                    x2 = np.random.randint(4)
                    y2 = np.random.randint(4)

                # initiating the positions of x1, y1, x2, y2
                p1, p2 = combinations.pop()
                x1 = int(p1/4)
                y1 = p1%4
                x2 = int(p2/4)
                y2 = p2%4
                print(x1,y1,x2,y2)
                x1 = 1
                y1 = 1
                x2 = 3
                y2 = 3
                winner, states, pis, zs, max_flag = start(x1,y1,x2,y2, max_iterations, model, itr) # one self-play until terminal
                
                if max_flag == False:
                    # if max_flag is true, it means that the game got stuck in the loop and reached the max number of iterations and didn't end, we skip these data.
                    for each in states:
                        print(each.hash())
                        dataset.append(create_dataset(each))
                        
                    for p in pis:
                        pi_list.append(p)
                    for z in zs:
                        z_list.append(z)

            
            if len(z_list) != 0: # this means that we have data pairs (states, pi, z)
                NP_dataset = np.array(dataset)
                NP_pi_list = np.array(pi_list)
                NP_z_list = np.array(z_list).reshape(len(z_list), 1)
                print(NP_dataset.shape)
                print(NP_pi_list.shape)
                print(NP_z_list.shape)


                history = model.fit_model(NP_dataset, NP_pi_list, NP_z_list, epochs=100, batch_size=16)
                model_json = model.model.to_json()
                with open("new_saved_model_%s.json"%itr, "w") as json_file:
                    json_file.write(model_json)
                model.model.save_weights("new_saved_model_%s.h5"%itr)
                print("Saved model to disk")
                print(model.model.summary())
                
            
            
            input("End of one chunck of data training, going for the next iteration!")

               

    #####################################
    #               TEST                #
    #####################################

    print("*****************\n\n")
    "Load the model"
    # itr = 1
    # load json and create model
    # itr = 5
    # json_file = open('CC/new_saved_model_%s.json'%itr, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("CC/new_saved_model_%s.h5"%itr)
    # model = loaded_model
    # print("Loaded model from disk")
    # # compile_model
    # loaded_model.compile(
    #         loss={'output_head1': 'categorical_crossentropy', 'output_head2': 'mean_squared_error'},
    #         optimizer='adam',
    #         metrics={'output_head1': 'accuracy', 'output_head2': 'mean_squared_error'}
    #     )
    # score = loaded_model.predict(data)

    b = Board()
    x1 = 1
    y1 = 1
    x2 = 2
    y2 = 2
    u2 = rangedUnit(1, 1, "R2", 2, 3)
    u5 = rangedUnit(2, 2, "R1", 1, 3)

    b.unitList.append(u2)
    b.unitList.append(u5)
    b.updatePlayerBasedList()
    b.update_board()
    b.Draw()

    gameOver = False
    t = 0
    dataset = []
    labels = []
    count = 0

    while(gameOver==False):
        for u in b.unitList:
            if u.ownerID == 2:
                data = create_dataset(b)
                # out = model.model.predict(data.reshape(1,4,4,4))
                out = model.model.predict(data.reshape(1,3,4,4)) # without hitpoints it wqas (1,3,4,4)
                P = out[0][0]
                v = out[1][0][0]
                print(P, v)
                print(v)
                print("+++++++")
                pool = u.allPossibleActions(b.unitList)
                print(pool)
                mask_array = np.zeros(21)
                for each in pool:
                    if each == "left":
                        mask_array[16] = 1
                    elif each == "right":
                        mask_array[17] = 1
                    elif each == "up":
                        mask_array[18] = 1
                    elif each == "down":
                        mask_array[19] = 1
                    elif each[:6] == "attack":
                        opponentName = each[7:]
                        opponentUnit = b.returnUnitbyName(b.unitList, opponentName)
                        pos = opponentUnit.x * 4 + opponentUnit.y
                        mask_array[pos] = 1
                P*=mask_array
                P/= sum(P)
                print(P)
                # action_selected = P.tolist().index(np.random.choice(P, p=P))
                action_selected = P.tolist().index(max(P))
                print("ACTION SELECTED", action_selected)
                if action_selected >=0 and action_selected <=15:
                    if u.canAttack(b.unitList[1]):    #u5 bood, bokon unitlist[1]
                        u.actionList.append(["attack", b.unitList[1].name])

                elif action_selected ==16:
                    u.actionList.append(["left", None])
                elif action_selected ==17:
                    u.actionList.append(["right", None])
                elif action_selected ==18:
                    u.actionList.append(["up", None])
                elif action_selected ==19:
                    u.actionList.append(["down", None])
                else:
                    pass
                    # u.actionList.append(["None", None])
                             
            else:
                u.attackClosest(b.unitList)

        print("ACTION SET/ P1-P2")
        print(u5.actionList)
        print(u2.actionList)

        print("\nLET'S EVALUATE\n\n")


        #evaluate
        for eachU in b.unitList:
            if eachU.actionList:
                action1 = eachU.actionList.pop(0)
                # print("FIRST ACTION of %s POPPED: "%eachU.name)
                if len(eachU.actionList) != 0:
                    AssertionError
                evaluate(action1, eachU, b, b.board)

        b.updatePlayerBasedList()
        b.update_board()
        b.Draw()

        if b.isOver():
            gameOver =True
            b.is_over = True

        for u in b.unitList:
            print(u.name, ": ", u.hitpoints, "\t", u.x,"\t", u.y, "\t", u.x*4 + u.y)

        # print(t)
        input("T=%s ended!\n\n"%t)
        t +=1
 
        input("end of round\n")