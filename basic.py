import gymnasium as gym
from utils import plot_learning_curve, make_env
from dqn_agent import DQNAgent
import numpy as np
import torch


eval_model_file = "eval_policy.pth"
next_model_file = "next_policy.pth"

env = make_env('PongNoFrameskip-v4')

observation = env.reset()

best_score = -np.inf
n_games = 250
# n_games = 1

agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                    input_dims=(env.observation_space.shape),
                    n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                    batch_size=32, replace=1000, eps_dec=1e-5,
                    chkpt_dir='models/', algo='DQNAgent',
                    env_name='PongNoFrameskip-v4')

fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
        + str(n_games) + 'games'
figure_file = 'plots/' + fname + '.png'

n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in range(n_games):
    terminated = truncated = False
    observation = env.reset()

    score = 0
    while not (terminated or truncated):
        action = agent.choose_action(observation)
        observation_, reward, truncated, terminated, info = env.step(action)
        score += reward

        agent.store_transition(observation, action,
                                reward, observation_, terminated or truncated)
        agent.learn()
        observation = observation_
        n_steps += 1

    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    print('episode: ', i,'score: ', score,
            ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
        'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

    if avg_score > best_score:
        # agent.save_models()
        print("saving model")
        torch.save(agent.q_eval.state_dict(), eval_model_file)
        torch.save(agent.q_next.state_dict(), next_model_file)

        print("key: ", agent.q_eval.state_dict()["fc2.weight"])

        best_score = avg_score

    eps_history.append(agent.epsilon)

x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps_array, scores, eps_history, figure_file)

# truncated = terminated = False

# score = 0
# while not (terminated or truncated):
#     action = agent.choose_action(observation)
#     observation_, reward, truncated, terminated, info = env.step(action)
#     score += reward

# print(score)
