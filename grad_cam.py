import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
import torch
from utils import make_env
from SmoothGradCAMplusplus.cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp
from SmoothGradCAMplusplus.utils.visualize import visualize, reverse_normalize
from SmoothGradCAMplusplus.utils.imagenet_labels import label2idx, idx2label
import matplotlib.pyplot as plt
from PIL import Image
import torch, os, PIL.Image, numpy
from matplotlib import cm

EVAL_MODEL_FILE = "eval_policy.pth"


def overlay(im1, im2, alpha=0.5):
    import numpy
    return PIL.Image.fromarray((
        numpy.array(im1)[...,:3] * alpha +
        numpy.array(im2)[...,:3] * (1 - alpha)).astype('uint8'))

def resize_and_crop(im, d):
    if im.size[0] >= im.size[1]:
        im = im.resize((int(im.size[0]/im.size[1]*d), d))
        return im.crop(((im.size[0] - d) // 2, 0, (im.size[0] + d) // 2, d))
    else:
        im = im.resize((d, int(im.size[1]/im.size[9]*d)))
        return im.crop((0, (im.size[1] - d) // 2, d, (im.size[1] + d) // 2))

def spec_size(size):
    if isinstance(size, int): dims = (size, size)
    if isinstance(size, torch.Tensor): size = size.shape[:2]
    if isinstance(size, PIL.Image.Image): size = (size.size[1], size.size[0])
    if size is None: size = (224, 224)
    return size

def rgb_heatmap(data, size=None, colormap='hot', amax=None, amin=None, mode='bicubic', symmetric=False):
    size = spec_size(size)
    print(size)
    mapping = getattr(cm, colormap)
    scaled = torch.nn.functional.interpolate(data[None, None], size=size, mode=mode)[0,0]
    if amax is None: amax = data.max()
    if amin is None: amin = data.min()
    if symmetric:
        amax = max(amax, -amin)
        amin = min(amin, -amax)
    normed = (scaled - amin) / (amax - amin + 1e-10)
    return PIL.Image.fromarray((255 * mapping(normed)).astype('uint8'))

if __name__=="__main__":
    env = make_env('PongNoFrameskip-v4')
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                        input_dims=(env.observation_space.shape),
                        n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                        batch_size=32, replace=1000, eps_dec=1e-5,
                        chkpt_dir='models/', algo='DQNAgent',
                        env_name='PongNoFrameskip-v4')

    model = agent.q_eval
    model.load_state_dict(torch.load(EVAL_MODEL_FILE))
    print(model)
    target_layer = model.conv2
    print(type(target_layer))

    n_games = 1
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
    

        size = 224
        im = np.einsum( 'ijk->jki', observation[:3])
        im = (im*255).astype(np.uint8)
        im = resize_and_crop(Image.fromarray(im), size)

        image_tensor = torch.tensor([observation],dtype=torch.float32).to('cuda')
        
    wrapped_model = SmoothGradCAMpp(model, target_layer, n_samples=25, stdev_spread=0.15)
    cam, idx = wrapped_model(image_tensor)
    cam = cam.reshape(cam.shape[2], cam.shape[3])
    print("cam: ", cam.shape)
    heatmapcam = overlay(im, rgb_heatmap(cam.cpu(), size = 224))
    plt.imshow(heatmapcam, alpha=0.5, cmap='jet')
    plt.show()
