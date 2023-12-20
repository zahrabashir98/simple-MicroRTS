
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

    def __init__(self, exploration_weight=1.5):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.childrenObjects = dict()
        self.Pi = dict()
        # self.terminal_set = set()

    def returnDist(self, node):
        # NodeHash = node.hash()
        if node.is_over == True:
            raise RuntimeError(f"choose called on terminal node {node}")
        visit_array = np.zeros(21)
        
        if node.nodeType == "max": #player2's turn to play (in my setting, player 2 starts) -> that'w why we have p2_actions
            for child in self.childrenObjects[node]:
                # input(child.p2_actions[0][0])
                print(self.Q[child])
                print(self.N[child])
                if child.p2_actions[0][0] == "attack":
                    for unit in node.unitList:
                        if unit.ownerID ==1: #!=player 2
                            pos = unit.x*4 + unit.y
                            visit_array[pos] = self.N[child]
                            if child.is_over == True:
                                visit_array[pos] = 1

                elif child.p2_actions[0][0] == "left":
                    visit_array[16] = self.N[child]
                    if child.is_over == True:
                        visit_array[16] = 1

                elif child.p2_actions[0][0] == "right":
                    visit_array[17] = self.N[child]
                    if child.is_over == True:
                        visit_array[17] = 1

                elif child.p2_actions[0][0] == "up":
                    visit_array[18] = self.N[child]
                    if child.is_over == True:
                        visit_array[18] = 1

                elif child.p2_actions[0][0] == "down":
                    visit_array[19] = self.N[child]
                    if child.is_over == True:
                        visit_array[19] = 1

                else:
                    visit_array[20] = 1
                    assert 10!=10
        
        elif node.nodeType == "min": #player1's turn to play -> p1_actions
            for child in self.childrenObjects[node]:
                
                if child.p1_actions[0][0] == "attack": 
                    for unit in node.unitList:
                        if unit.ownerID ==2:
                            pos = unit.x*4 + unit.y
                            visit_array[pos] = self.N[child]
                            if child.is_over == True:
                                visit_array[pos] = 1
                
                elif child.p1_actions[0][0] == "left":
                    visit_array[16] = self.N[child]
                    if child.is_over == True:
                        visit_array[16] = 1

                elif child.p1_actions[0][0] == "right":
                    visit_array[17] = self.N[child]
                    if child.is_over == True:
                        visit_array[17] = 1

                elif child.p1_actions[0][0] == "up":
                    visit_array[18] = self.N[child]
                    if child.is_over == True:
                        visit_array[18] = 1
                        
                elif child.p1_actions[0][0] == "down":
                    visit_array[19] = self.N[child]
                    if child.is_over == True:
                        visit_array[19] = 1
                else:
                    visit_array[20] = 1
                    assert 11!=11

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
        # return best_board.p2_actions

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
        index = np.argmax(dist)
        # print(index)
        action_string = self.returnActionString(index)
        if node.nodeType == "max":
            for child in self.childrenObjects[node]:
                if child.p2_actions[0][0] == action_string:
                    return child 
    
        elif node.nodeType == "min":
            for child in self.childrenObjects[node]:
                if child.p1_actions[0][0] == action_string:
                    return child 
        

    def do_rollout(self, node, model, VanillaMCTS):
        "Make the tree one layer better."
        """
        if you don't wanna run vannilla MCTS, change the flag (it is called in the game.py)!
        """
        root = node
        # self.N[root] = 1 #BUG???
        path = self._select(node, VanillaMCTS)
        leaf = path[-1]

        if VanillaMCTS:
            self._expandd(leaf)
            reward = self._simulate(leaf)
        else:
            reward = self._expand(leaf, model)

        
        if reward != None: # if already expanded, I'm still adding to the count and val.
            self._backpropagate(path, reward, leaf)
        else:
            print(reward)
            AssertionError
        # print("******************\n")


    def _select(self, node, vMCTSFlag):
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

            # descend a layer deeper
            if vMCTSFlag: #Naive MCTS UCT
                node = self._uct_select_naive(node)
            else: #Alpha Zero UCT
                node = self._uct_select(node)  
   
    
    def mask_probability_vector(self, allIndexes, P):
        mask = np.zeros_like(P)
        mask[allIndexes] = 1
        masked_probabilities = P * mask
        normalization_factor = np.sum(masked_probabilities)
        normalized_probabilities = masked_probabilities / normalization_factor
        return normalized_probabilities


    def create_dataset(self, board):
        data = np.zeros((2, 4, 4))
        # fourth_layer = np.zeros((1, 4, 4))
        if board.nodeType == "max":
            for eachU in board.unitList:
                if eachU.ownerID == 2:
                    data[0][eachU.x][eachU.y] = 1
                    # fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints
                else:
                    data[1][eachU.x][eachU.y] = 1
                    # fourth_layer[0][eachU.x][eachU.y] = eachU.hitpoints

            third_layer = np.zeros((1,4,4)) #Player turn to play - o for p2 (player1)
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
    

    def _expandd(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.childrenObjects:
            return  # already expanded
            
        childen, _, _  = node.find_children()
        self.childrenObjects[node] = childen


    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        # print("begin simulate")
        while True:
            if node.isOver() == True:
                if node.winner == 0:
                    AssertionError
                    print("shouldn't happen")
                elif node.winner ==2:
                    return 1
                elif node.winner == 1:
                    return -1
                elif node.winner == 0.5:
                    return 0
                else:
                    AssertionError
                    return None
                # return 1 - reward if invert_reward else reward #TODO: how to assign reward in Realtime games?
            node = node.find_random_child()

 
    def _expand(self, node, model):
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
                    value = 0
                print(value)
                return value
            assert 14!=14
            return
    

        if node.is_over == True:
            if node.winner == 2: #player2 in my setting is the first who plays :D
                value = 1
            if node.winner == 1:
                value = -1
            if node.winner == 0.5:
                value = 0
            self.childrenObjects[node] = []
            print(value)
            return value 
        

        data = self.create_dataset(node)
        # data = np.array(data).reshape(1, 4, 4, 4) 
        data = np.array(data).reshape(1, 3, 4, 4)
        ### get prediction from model ###
        output = model.model.predict(data)
        p = output[0][0]
        v = output[1][0][0]
        # print(p)

        try:
            childernObjs, _, allIndexes = node.find_children()
        except:
            print(node.hash())
            AssertionError
        
        if allIndexes == []:
            print(p)
            print(node.hash())
            AssertionError
        
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
        if leaf.winner == 0.5:
            print("IN LEAF WINNER  == 0.5")
            for node in reversed(path): #CHANGED
                if node.nodeType == "max":
                    self.N[node] += 1
                    self.Q[node] += -0.5
                elif node.nodeType == "min":
                    self.N[node] += 1
                    self.Q[node] += 0.5
        else:
            for node in reversed(path):
                self.N[node] += 1
                self.Q[node] += reward

                
            # reward *= -1 
    

    def _uct_select_naive(self, node):
        "Select a child of node for Vanilla MCTS"
        scores = []
        if node.nodeType =="max":
            for child in self.childrenObjects[node]:
                log_N_vertex = math.log(self.N[node])
                scores.append(self.Q[child] / self.N[child] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[child]))
                # print(self.Q[child] / self.N[child] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[child]))
            max_index = scores.index(max(scores))
            # print(self.childrenObjects[node][max_index].p2_actions)
            return self.childrenObjects[node][max_index]

        elif node.nodeType == "min":
            for child in self.childrenObjects[node]:
                log_N_vertex = math.log(self.N[node])
                scores.append(self.Q[child] / self.N[child] - self.exploration_weight * math.sqrt(log_N_vertex / self.N[child]))
                # print(self.Q[child] / self.N[child] - self.exploration_weight * math.sqrt(log_N_vertex / self.N[child]))
            min_index = scores.index(min(scores))
            # print(self.childrenObjects[node][min_index].p1_actions)
            return self.childrenObjects[node][min_index]


    def _uct_select(self, node):
        "Select a child of node, for alpha-zero, balancing exploration & exploitation"
        scores = []

        if node.nodeType =="max":
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



        elif node.nodeType == "min":
            for child in self.childrenObjects[node]:
                if self.N[child] == 0:
                    q_value = 0
                else:
                    q_value  = (self.Q[child] / self.N[child] )
                scores.append(q_value - (self.exploration_weight * (math.sqrt(self.N[node]) / (self.N[child] + 1)) * self.Pi[child]))
                # print("score", q_value - (self.exploration_weight * (math.sqrt(self.N[node]) / (self.N[child] + 1)) * self.Pi[child]))
                # print("---")
            min_index = scores.index(min(scores))
            return self.childrenObjects[node][min_index]


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
