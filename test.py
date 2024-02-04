import gymnasium as gym
from utils import plot_learning_curve, make_env
from dqn_agent import DQNAgent
import numpy as np
import gymnasium as gym
from utils import plot_learning_curve, make_env
from dqn_agent import DQNAgent
import numpy as np
import torch

eval_model_file = "eval_policy.pth"

env = make_env('PongNoFrameskip-v4')

observation = env.reset()

best_score = -np.inf
n_games = 20

agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                    input_dims=(env.observation_space.shape),
                    n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                    batch_size=32, replace=1000, eps_dec=1e-5,
                    chkpt_dir='models/', algo='DQNAgent',
                    env_name='PongNoFrameskip-v4')

# agent.load_models()


model = agent.q_eval
model.load_state_dict(torch.load(eval_model_file))

print("key: ", model.state_dict()["fc2.weight"])

n_games = 20
scores = []
frames = []

for i in range(n_games):
    terminated = truncated = False
    observation = env.reset()
    
    score = 0
    n_steps = 0
    while not (terminated or truncated):
        state = torch.tensor([observation],dtype=torch.float).to('cuda')
        actions = model(state)
        action = torch.argmax(actions).item()
        observation_, reward, terminated, truncated, info = env.step(action)
        score += reward
        observation = observation_
        n_steps += 1
        if n_steps%10 == 0:
            frames.append(observation)
        if n_steps == 500:
            break
    scores.append(score)
    print('episode: ', i,'score: ', score,
        'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
    