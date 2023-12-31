
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from random import choice
import numpy as np
import random
from NN import *

# np.random.seed(1)
# random.seed(1)
class MCTS:

    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=5):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.childrenObjects = dict()
        self.Pi = dict()
        # self.terminal_set = set()
    
    def returnUnitbyName(self, unitList, name): #TODO: make sure we don't have duplicate names
        for u in unitList:
            if u.name == name:
                return u
        
    def returnDist(self, node):
        # NodeHash = node.hash()
        if node.is_over == True:
            raise RuntimeError(f"choose called on terminal node {node}")
        visit_array = np.zeros(21)
        
        for child in self.childrenObjects[node]:

            print(child.single_unit_action)
            print(self.Q[child])
            print(self.N[child])
            # input()

            if child.single_unit_action[:6] == "attack":
                targetName = child.single_unit_action[7:]
                targetUnit = self.returnUnitbyName(node.unitList, targetName)
                pos = targetUnit.x*4 + targetUnit.y
                visit_array[pos] = self.N[child]
                # if child.is_over == True:
                #     visit_array[pos] = 1

            elif child.single_unit_action == "left":
                visit_array[16] = self.N[child]
                # if child.is_over == True:
                #     visit_array[16] = 1

            elif child.single_unit_action == "right":
                visit_array[17] = self.N[child]
                # if child.is_over == True:
                #     visit_array[17] = 1

            elif child.single_unit_action == "up":
                visit_array[18] = self.N[child]
                # if child.is_over == True:
                #     visit_array[18] = 1

            elif child.single_unit_action == "down":
                visit_array[19] = self.N[child]
                # if child.is_over == True:
                #     visit_array[19] = 1

            else:
                visit_array[20] = 1
                assert 10!=10
        print("ARRAY E N")
        print(visit_array)
        try:
            visit_array /= sum(visit_array)
        except:
            print(visit_array)
            print(node.hash())
            print(self.childrenObjects[node])
            input("Shouldn't have happened")
            assert 12!=12
        return visit_array
        
    def choose(self, node):
        "Choose the best successor of node -> Max(val/n). (Choose a move in the game)"


        if node.is_over == True:
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.childrenObjects:
            input("HEEERREEE")
            assert 3!=3
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        count = 0
        for each in self.childrenObjects[node]:
            if score(each) == float("-inf"):
                count += 1
        
        if node.nodeType == "max":
            best_board = max(self.childrenObjects[node], key=score)
        else:
            best_board = max(self.childrenObjects[node], key=score)

        if count == len(self.childrenObjects[node]):
            assert 20!=20
        
        return best_board


    def returnActionString(self, index):
        if index>=0 and index<=15:
            action_string = "attack"
        elif index == 16:
            action_string = "left"
        elif index == 17:
            action_string = "right"
        elif index == 18:
            action_string = "up"
        elif index == 19:
            action_string = "down"
        elif index == 20:
            action_string = "nothing"

        return action_string


    def choose_based_on_dist(self, node, dist):
        "This function chooses the next step of the game but only based on the maximum number which has been visited."
        print("*****\n")
        index = np.argmax(dist)
        action_string = self.returnActionString(index)
        # print(index)
        # print(action_string)

        for child in self.childrenObjects[node]:
            # print(child.single_unit_action)
            if child.single_unit_action == action_string:
                return child 
            elif child.single_unit_action[:6] == action_string:
                return child 

        assert 1000!=1000

    

    def do_rollout(self, node, model, mainUnitName): #TODO: change rollout for one unitname
        "Make the tree one layer better."
        """
        if you don't wanna run vannilla MCTS, change the flag (it is called in the game.py)!
        """
        root = node
        # self.N[root] = 1 #BUG???
        path = self._select(node)
        print(path)
        
        leaf = path[-1]
        # print(leaf.hash())
        reward = self._expand(leaf, model, mainUnitName)
        if reward ==1: #or reward ==-1 or reward ==0.5
            print("THIS:")
            print(reward)
            for each in path:
                print(each.single_unit_action)
            print("------")
            # input()
        
        if reward != None: # if already expanded, I'm still adding to the count and val.
            self._backpropagate(path, reward, leaf)
        else:
            print(reward)
            AssertionError
        # print("******************\n")


    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.childrenObjects or not self.childrenObjects[node]:
                return path
        
            unexplored = self.childrenObjects[node] - self.childrenObjects.keys() #- self.terminal_set

            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

            node = self._uct_select(node)  
   
    
    def mask_probability_vector(self, allIndexes, P):
        mask = np.zeros_like(P)
        mask[allIndexes] = 1
        masked_probabilities = P * mask
        normalization_factor = np.sum(masked_probabilities)
        normalized_probabilities = masked_probabilities / normalization_factor
        return normalized_probabilities


    def create_dataset(self, board, unitName):
        data = np.zeros((2, 4, 4))
        third_layer = np.zeros((1, 4, 4))
        fourth_layer = np.zeros((5, 4, 4))

        for eachU in board.unitList:
            if eachU.ownerID == 2:
                data[0][eachU.x][eachU.y] = 1
                fourth_layer[eachU.hitpoints-1][eachU.x][eachU.y] = 1
            else:
                data[1][eachU.x][eachU.y] = 1
                fourth_layer[eachU.hitpoints-1][eachU.x][eachU.y] = 1
        
        for eachU in board.unitList: # TODO: (to discuss) I designed it in a way that says, it is this unit's turn (not player)
            if eachU.name == unitName:
                third_layer[0][eachU.x][eachU.y] = 1
        
        data = np.concatenate((data, third_layer, fourth_layer))

        return data
    

    def _expand(self, node, model, mainUnitName):
        "Update the 'children' dict with the children of 'node'"

        if node in self.childrenObjects: #Should I update each time we see an already expanded node? (terminal)
            print("ALREADY EXPANDED\n") 
            print(self.childrenObjects[node]) # only happens in the terminal nodes becuase they don't have children in the tree, and they'll be considered unexplored.
            if node.is_over == True:
                if node.winner == 2:
                    value = 1
                elif node.winner == 1:
                    value = -1
                elif node.winner == 0.5:
                    value = 0.5
                print(value)
                return value
            assert 14!=14
            return
    

        if node.is_over == True:
            print("IS OVER")
            print(node.hash())
            print(node.winner)

            if node.winner == 2: #player2 in my setting is the first who plays :D #TODO: change it to mainplayerID
                value = 1
            if node.winner == 1:
                value = -1
            if node.winner == 0.5:
                value = 0.5
            self.childrenObjects[node] = []
            print(value)
            # input()
            return value 
        
        # who wants to play => R2 => to get the ps for expansion
        data = self.create_dataset(node, mainUnitName)
        print(data)
        data = np.array(data).reshape(1, 8, 4, 4)

        ### get prediction from model ###
        output = model.model.predict(data)
        p = output[0][0]
        v = output[1][0][0]
        # print(p)

        try:
            print("SHORO")
            childernObjs, allIndexes = node.find_children(mainUnitName)
            # print(childernObjs)
            print(allIndexes)
        except AssertionError as msg:
            print(node.hash())
            print(msg)
            assert 10!=10
            
        
        if allIndexes == []:
            print(p)
            print(node.hash())
            assert 11!=11
        
        newP = np.array(self.mask_probability_vector(allIndexes, p))
        newP = newP[allIndexes]
        print(newP)
        self.childrenObjects[node] = childernObjs

        i = 0
        for each in childernObjs:
            self.Pi[each] = newP[i]
            i += 1
        print("CHILDREN FOUND and Ps assigned")
        return v
        

    def _backpropagate(self, path, reward, leaf):
        "Send the reward back up to the ancestors of the leaf"
        # if leaf.winner == 0.5:
        #     for node in reversed(path): #CHANGED
        #         if node.nodeType == "max":
        #             self.N[node] += 1
        #             self.Q[node] += -0.5
        #         elif node.nodeType == "min":
        #             self.N[node] += 1
        #             self.Q[node] += 0.5
        # else:
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

                
            # reward *= -1 


    def _uct_select(self, node):
        "Select a child of node, for alpha-zero, balancing exploration & exploitation"
        scores = []
        for child in self.childrenObjects[node]:
            if self.N[child] == 0:
                q_value = 0
            else:
                q_value  = (self.Q[child] / self.N[child] )
            scores.append(q_value + (self.exploration_weight * (math.sqrt(self.N[node]) / (self.N[child] + 1)) * self.Pi[child]))
            # print("score", q_value + (self.exploration_weight * (math.sqrt(self.N[node]) / (self.N[child] + 1)) * self.Pi[child]))
            # print("---")
        max_index = scores.index(max(scores))
        return self.childrenObjects[node][max_index]



class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
