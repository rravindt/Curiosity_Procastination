"""This is a simple implementation of [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)"""

import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import argparse


class Curiosity:

    def __init__(self, numberOfActions, dimStates):
        self.numberOfActions = numberOfActions
        self.dimStates = dimStates
        self.batchSize = 128
        self.learningRate = 0.01
        self.gamma = 0.95
        self.epsilon = 0.95
        self.memorySize = 10000
        self.minMemorySize = 500
        self.targetUpdateCounter = 0
        self.targetUpdateThreshold = 10
        self.inverseNetUpdateThreshold = 1000
        self.inverseNetUpdateCounter = 0


        self.memory = deque(maxlen = self.memorySize)
        self.policyNet = self.createPolicyNet()
        self.targetPolicyNet = self.createPolicyNet()
        self.targetPolicyNet.set_weights(self.policyNet.get_weights())
        self.inverseNet = self.createInverseNet()



    def createPolicyNet(self):
        model = Sequential()
        model.add(Dense(64, input_dim = self.dimStates, kernel_initializer = "uniform", activation = "relu"))
        model.add(Dense(self.numberOfActions))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learningRate), metrics = ['accuracy'])
        return model


    def createInverseNet(self):
        inputSize = self.dimStates + 1
        model = Sequential()
        model.add(Dense(64, input_dim = inputSize, kernel_initializer = "uniform", activation = "relu"))
        model.add(Dense(self.dimStates))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learningRate), metrics = ['accuracy'])
        return model


    def updateMemory(self, transition):
        self.memory.append(transition)


    def getAction(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = np.argmax(self.policyNet.predict(np.array(state).reshape(-1, len(state))))
        else:
            action = np.random.randint(0, self.numberOfActions)
        return action


    def getInverseNetPredictionError(self, states, actions):
        inputToInverseNet = np.concatenate((states, actions.reshape(self.batchSize, 1)), axis = 1)
        predictedNextStates = self.inverseNet.predict(inputToInverseNet)
        temp = np.array(states) - np.array(predictedNextStates)
        return np.mean(temp**2, axis = 1)



    def train(self, isTerminalState):
        if(len(self.memory) < self.minMemorySize):
            return

        minibatch = random.sample(self.memory, self.batchSize)

        currentStates = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        currentQList = self.targetPolicyNet.predict(currentStates)

        newStates = np.array([transition[3] for transition in minibatch])
        futureQList = self.policyNet.predict(newStates)

        intrinsicReward = self.getInverseNetPredictionError(currentStates, actions)

        X = []
        Y = []


        for index, (currentState, action, reward, newState, done) in enumerate(minibatch):
            # print(intrinsicReward[index])
            if not done:
                maxFutureQ = np.max(futureQList[index])
                newQ = intrinsicReward[index] + reward + self.gamma*maxFutureQ
            else:
                newQ = intrinsicReward[index] + reward


            currentQs = currentQList[index]
            currentQs[action] = newQ

            X.append(currentState)
            Y.append(currentQs)



        self.policyNet.fit(np.array(X), np.array(Y), batch_size = self.batchSize, verbose = 0, shuffle = False)

        if isTerminalState:
            self.targetUpdateCounter += 1
        self.inverseNetUpdateCounter += 1


        if(self.targetUpdateCounter % self.targetUpdateThreshold == 0):
            self.targetPolicyNet.set_weights(self.policyNet.get_weights())
            self.targetUpdateCounter = 0

        if(self.inverseNetUpdateCounter % self.inverseNetUpdateThreshold == 0):
            self.inverseNet.fit(np.concatenate((currentStates, actions.reshape(self.batchSize, 1)), axis = 1), newStates, batch_size = self.batchSize, verbose = 0, shuffle = False)

def getStateDiff(a):
    return (abs(a[0]-0.5))


def introduceNoisyTV(state, tvIsOn):
    if(tvIsOn):
        state = np.append(state, [random.uniform(-2, 2)])
    else:
        state = np.append(state, [0])
    return state

env = gym.make('MountainCar-v0')
env = env.unwrapped

parser = argparse.ArgumentParser()
parser.add_argument("--noisy", default=0, type=int)
args = parser.parse_args()
curios_model = args.noisy
while curios_model > 2:
    curios_model = int(input("Enter a number between 0-2"))
curiosity = Curiosity(3, 3)
ep_steps = []
tvFlag =  False
for epi in range(1, 101):
    s = env.reset()
    steps = 1
    while True:
        env.render()      #This line can be commented or uncommented for the visual running of the curiosity model
        if(len(s) == 2):
            s = introduceNoisyTV(s, False)
        else:
            s[2] = 0
        a = curiosity.getAction(s)
        s_, r, done, info = env.step(a)
        if(epi >= 25 and epi < 75):
            if curios_model == 1:
                s_[1] = random.uniform(-2, 2)
                s_[0] = random.uniform(-2, 2)
            if curios_model == 2: tvFlag = True
        
        if done:
            r = 10
        else: r = 0
        
        
        # Storing in Memory
        if(tvFlag):
            s_ = introduceNoisyTV(s_, True)
        else:
            s_ = introduceNoisyTV(s_, False)
        curiosity.updateMemory((s, a, r, s_, done))
        curiosity.train(done)
        if done:
            #r=r+22
            print('Epi: ', epi, "| steps: ", steps)
            ep_steps.append(steps)
            break
        s = s_
        steps += 1


with open('foo.pkl', 'wb') as f:
    pickle.dump(ep_steps, f)


print('Average Steps: ' + str(np.mean(ep_steps)))
print('Average Steps During Noisy TV: ' + str(np.mean(ep_steps[25:75])))
plt.plot(ep_steps)
plt.ylabel("steps")
plt.xlabel("episode")
plt.show()

