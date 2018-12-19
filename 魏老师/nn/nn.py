# -*- coding: utf-8 -*-

def sigmoid(x):
    temp = np.exp(x)
    return temp/(1+temp)


def sigmoid_1(x):
    temp = np.exp(x)
    return temp/np.power(1+temp,2)

class Network:
    def __init__(self,num_nodes):
        self.num_nodes = num_nodes
        self.num_layers = len(num_nodes) - 1
        self.W = [None]*self.num_layers
        self.b = [None]*self.num_layers
        for i in range(self.num_layers):
            self.W[i]=np.random.randn(num_nodes[i+1],num_nodes[i])
            self.b[i]=np.random.randn(num_nodes[i+1],1)



    def initialization(self,a,b):
        return np.random.rand([a,b])


