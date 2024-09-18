import gym
import keyboard
import time

env = gym.make("SnakeGame-v0", render_mode="human")

env.reset()

step = 0
episode = 0
while True:
    action = env.action_space.sample()
    # if keyboard.is_pressed('up'): #Forward
    #     action = 0
    # elif keyboard.is_pressed('right'): #Right
    #     action = 2
    # elif keyboard.is_pressed('left'): #Left
    #     action = 1
    # elif keyboard.is_pressed('q'):
    #     break
    # else:
    #     time.sleep(0.1)
    
    if action is not None:
        next_state, reward, done, info = env.step(action)
        step += 1
        env.render(options={'episode_index': episode, 'step': step})

        info = env.get_full_state()
        print(info['hp'])
        
        if done:
            print(f"Episode {episode} finished after {step} steps")
            env.reset()
            episode += 1
            step = 0