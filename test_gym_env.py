import gym
import sys
import time
import msvcrt
getch = msvcrt.getch

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78)



env = gym.make('FrozenLakeNotSlippery-v0')
env
env.reset()
env.render()

while True:
    action = None
    
    key = getch()
    
    if key == b'w':
        action = UP
    elif key == b's':
        action = DOWN  
    elif key == b'a':
        action = LEFT
    elif key == b'd':
        action = RIGHT
    
    if key == b'r':
        state, reward, done, prob = env.reset(), 0, 0, {'prob': 0.0}
        env.render()
    
    if key == b'q':
        sys.exit(0)
    
    if isinstance(action, int):
        state, reward, done, prob = env.step(action)
        print('current state: {}'.format(state))
        print('current reward: {}'.format(reward))
        print('current done: {}'.format(done))
        print('current prob: {}'.format(prob))
        env.render()
    
    if done:
        if reward:
            print('You Win!!')
        else:
            print('You Loose!!')
    