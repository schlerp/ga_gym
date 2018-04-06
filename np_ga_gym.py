import numpy as np
import gym
import random
import copy
import multiprocessing

from np_networks import *


POP_SIZE = 50
NET_LAYERS = 3
NET_WIDTH = 16
EPOCHS = 10
TOP_K = 10
THREADS = 8

RENDER_ENV = False

GAME = 'Taxi-v2'

#GAME = 'FrozenLakeNotSlippery-v0'
#from gym.envs.registration import register
#register(
    #id=GAME,
    #entry_point='gym.envs.toy_text:FrozenLakeEnv',
    #kwargs={'map_name' : '4x4', 'is_slippery': False},
    #max_episode_steps=100,
    #reward_threshold=0.78)


def uncompress_network(compressed_network, input_size, output_size):
    seed = compressed_network.seed
    other_seeds = compressed_network.other_seeds
    network = GeneticNetwork1D(input_size, output_size, seed, NET_WIDTH, NET_LAYERS)
    network.compile()
    for new_seed in other_seeds:
        network.evolve(new_seed)
    return network


class Agent(object):
    def __init__(self, compressed_net, env):
        self.network = uncompress_network(compressed_net, 
                                          env.observation_space.n, 
                                          env.action_space.n)
        self.network.compile()
        self.env = env
        self.total_reward = 0
        
    def run(self, render=False):
        state, reward, done, prob = self.env.reset(), 0, False, {'prob': 0.0}
        if render:
            self.env.render()
            
        while not done:
            state = one_hot(state, env.observation_space.n)
            action = self.network.forward(state)
            state, reward, done, prob = self.env.step(action)
            self.total_reward += reward
        return (self.total_reward, self.network.seed, self.network.other_seeds)
        

def one_hot(x, classes):
    '''returns onehot encoded vector for each item in x'''
    temp = [0. for _ in range(classes)]
    temp[int(x)] = 1.
    return np.array(temp)

if __name__ == '__main__':
    
    print('creating initial agents...')
    agents = []
    
    envs = [gym.make(GAME) for i in range(POP_SIZE)]
    
    for i in range(POP_SIZE):
        env = envs[i]
        net = CompressedNetwork()
        agents.append(Agent(net, env))
    
    #pool = multiprocessing.Pool(THREADS)
    
    
    for current_epoch in range(EPOCHS):
        print('running epoch {}...'.format(current_epoch+1))
        
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(THREADS)        
        
        results = []
        for i, agent in enumerate(agents):
            #print('running agent: {}'.format(i))
            results.append(pool.apply_async(agent.run))
        
        pool.close()
        pool.join()
        
        results = [r.get() for r in results]
        
        
        print('selecting fittest agents...')
        results.sort(key=lambda x: x[0], reverse=True)
        fittest_agents = results[0:TOP_K]
        
        print('top agent scores: {}'.format([x[0] for x in fittest_agents]))
        print('top agent other seeds length: {}'.format(len(fittest_agents[0][2])))
        print('top agent other seeds: {}'.format(fittest_agents[0][2]))
        
        if current_epoch <= EPOCHS-1:
            print('creating next generation...')
            agents = []
            for i in range(POP_SIZE):
                env = envs[i]
                _, seed, other_seeds = random.choice(fittest_agents)
                new_seed = random.randint(0, 2**8-1)
                other_seeds = copy.copy(other_seeds)
                other_seeds.append(new_seed)
                net = CompressedNetwork(seed, other_seeds)
                agents.append(Agent(net, env))
        else:
            print('best agent:')
            print('  reward: {}'.format(results[0][0]))
            print('  seed: {}'.format(results[0][1]))
            print('  other seeds: {}'.format(results[0][2]))
        
        print('\n')
        