from keras import layers
import numpy as np
import pandas as pd

class nn_config:
    def __init__(self, layers, ipnodes, tnodes, activation):
        self.layers = layers
        self.ipnodes = ipnodes
        self.tnodes = tnodes
        self.activation = activation
        
    def node_list_gen(self):
        nominal_nodes = np.floor([tn/lr for tn, lr in zip(self.tnodes, self.layers)])
        nl = []
        for n, l in zip(nominal_nodes, self.layers):
            nl.append([n]*l)
        self.node_list = nl
        return self
    
    def nn_architecture(self, hlayers, nodes, activation):
        conf = {'hlayers': hlayers,
                'ipnodes': self.ipnodes,
                'nodes': nodes,
                'activation': activation*hlayers
                }
        return conf
    
    def creat_config(self):        
        config = []
        self.node_list_gen()
        for l, n in zip(self.layers, self.node_list):
            conf = self.nn_architecture(l, n, self.activation)
            config.append(conf)
        return config