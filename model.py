import pickle
from snake import Snake, Direction
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
import random


"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""

def prepare_data(file_path):
    with open(file_path, 'rb') as f:
        data_file = pickle.load(f)
    
    inputs = np.empty((len(data_file['data']),8))
    outputs = np.empty(len(data_file['data']))

    prev_len_body = 0
    
    i=0

    for game_state in data_file['data']:

        len_body = len(game_state[0]['snake_body'])
        if len_body >= prev_len_body:
            prev_len_body = len_body
        else:
            inputs = inputs[:-1]
            outputs = outputs[:-1]
            prev_len_body = 0
            i -= 1
        
        inputs[i] = game_state_to_data_sample(game_state[0], data_file["bounds"], data_file["block_size"])
        outputs[i] = game_state[1].value
        i += 1

    return inputs, outputs


def check_if_bound(head, bounds, attributes):
    if head[0] == 0:
        attributes['PL'] = 1
    if head[0] == bounds[0]:
        attributes['PP'] = 1
    if head[1] == 0:
        attributes['PG'] = 1
    if head[0] == bounds[0]:
        attributes['PD'] = 1
    return attributes


def check_if_body(head, body, attributes, block_size):
    for part in body:
        if (part[0] + block_size, part[1]) == head:
            attributes['PL'] = 1
        if (part[0] - block_size, part[1]) == head:
            attributes['PP'] = 1
        if (part[0], part[1] + block_size) == head:
            attributes['PG'] = 1
        if (part[0], part[1] - block_size) == head:
            attributes['PD'] = 1
    return attributes  


def check_where_food(head, food, attributes):
    if head[0] > food[0]:
        attributes['FL'] = 1
    if head[0] < food[0]:
        attributes['FP'] = 1
    if head[1] > food[1]:
        attributes['FG'] = 1
    if head[1] < food[1]:
        attributes['FD'] = 1
    return attributes


def game_state_to_data_sample(game_state: dict, bounds, block_size):
    attributes = {
        'PD': 0,
        'PG': 0,
        'PL': 0,
        'PP': 0,
        'FG': 0,
        'FD': 0,
        'FL': 0,
        'FP': 0
    }

    head = game_state['snake_body'][-1]

    attributes = check_if_bound(head, bounds, attributes)

    attributes = check_if_body(head, game_state['snake_body'], attributes, block_size)

    attributes = check_where_food(head, game_state['food'], attributes)

    return list(attributes.values())


class Node:
    def __init__(self, attribute=None, decision=None, child_node_true=None, child_node_false=None):
        self.attribute = attribute
        self.action = decision
        self.child_node_true = child_node_true
        self.child_node_false = child_node_false
    
    def decision(self, game_state):
        if self.action is not None:
            return self.action
        elif self.attribute is None and self.action is None:
            return None
        elif game_state[self.attribute] == True:
            return self.child_node_true.decision(game_state)
        elif game_state[self.attribute] == False:
            return self.child_node_false.decision(game_state)

def prepare_probability(path="data/snake.pickle"):
    # Estymacja prawdopodobieństw
    inputs, outputs = prepare_data(path)
    l = len(outputs)
    alpha = 0
    ratio = 0.25

    inputs = inputs[:int(l*ratio)]
    outputs = outputs[:int(l*ratio)]

    probs = np.zeros((4, 8, 2))
    p = np.zeros(4)
    
    for i in range(4):
        p[i] = (outputs == i).sum() / len(outputs) # Prawdopodobieństwo a priori
        for j in range(8):
            for k in range(2):
                probs[i, j, k] = (((inputs[:, j] == k) & (outputs == i)).sum()+alpha) / ((outputs == i).sum()+ 2*alpha) # Prawdopodobieństwa  wartości dla pojedynczych atrybutów
    
    return p, probs

def predict_proba(x, p_priori, probs):
    # Predykcja
    p = np.ones(4)
    for i in range(4):
        p[i] = p_priori[i]
        for j in range(8):
            p[i] *= probs[i, j, int(x[j])]
    return p / p.sum()

def predict(x, p_priori, probs):
    return np.argmax(predict_proba(x, p_priori, probs))


if __name__ == "__main__":
    """ Example of how to read a pickled file, feel free to remove this"""
    path="data/snake.pickle"
    inputs, outputs = prepare_data(path)
    l = len(outputs)
    ratio = 0.25

    p_priori, probs = prepare_probability()

    train_inputs = inputs[:int(l*ratio)]
    train_outputs = outputs[:int(l*ratio)]

    val_inputs = train_inputs[int(0.8*len(train_inputs)):]
    val_outputs = train_outputs[int(0.8*len(train_inputs)):]

    counter = 0

    for i in range(len(val_inputs)):
        action = predict(val_inputs[i], p_priori, probs)

        if action == val_outputs[i]:
            counter += 1

    counter_train = 0

    for i in range(len(train_inputs)):
        action = predict(train_inputs[i], p_priori, probs)

        if action == train_outputs[i]:
            counter_train += 1

    print('Validation accuracy: ', counter/len(val_outputs))

    print('Train accuracy: ', counter_train/len(train_outputs))


