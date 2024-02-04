import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
import matplotlib.pyplot as plt
import torch

import torch, os, PIL.Image, numpy
from matplotlib import cm
import numpy as np
from PIL import Image
from collections import OrderedDict
# from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop

import torch
import contextlib
from baukit.baukit import show, renormalize, pbar

from SmoothGradCAMplusplus.cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp
from SmoothGradCAMplusplus.utils.visualize import visualize, reverse_normalize
from SmoothGradCAMplusplus.utils.imagenet_labels import label2idx, idx2label

EVAL_MODEL_FILE = "eval_policy.pth"


class Trace(contextlib.AbstractContextManager):

    def __init__(
        self,
        module,
        layer=None):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(m, inputs, output):
            print("output: ", output.shape)
            retainer.output = recursive_copy(
                output
            )
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = False
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
        
    def close(self):
        self.registered_hook.remove()


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)
def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        print("---5---")
        return x
    if isinstance(x, torch.Tensor):
        print("---1---")
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        print("2")
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        print("3")
        return type(x)([recursive_copy(v) for v in x])
    else:
        print("4")
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


# Helper Functions
def resize_and_crop(im, d):
    if im.size[0] >= im.size[1]:
        im = im.resize((int(im.size[0]/im.size[1]*d), d))
        return im.crop(((im.size[0] - d) // 2, 0, (im.size[0] + d) // 2, d))
    else:
        im = im.resize((d, int(im.size[1]/im.size[9]*d)))
        return im.crop((0, (im.size[1] - d) // 2, d, (im.size[1] + d) // 2))

def overlay(im1, im2, alpha=0.5):
    import numpy
    return PIL.Image.fromarray((
        numpy.array(im1)[...,:3] * alpha +
        numpy.array(im2)[...,:3] * (1 - alpha)).astype('uint8'))

def rgb_threshold(data, size=None, mode='bicubic', p=0.2):
    size = spec_size(size)
    scaled = torch.nn.functional.interpolate(data[None, None], size=size, mode=mode)[0,0]
    ordered = scaled.view(-1).sort()[0]
    threshold = ordered[int(len(ordered) * (1-p))]
    result = numpy.tile((scaled > threshold)[:,:,None], (1, 1, 3))
    return PIL.Image.fromarray((255 * result).astype('uint8'))

def overlay_threshold(im1, im2, alpha=0.5):
    import numpy
    return PIL.Image.fromarray((
        numpy.array(im1)[...,:3] * (1 - numpy.array(im2)[...,:3]/255) * alpha +
        numpy.array(im2)[...,:3] * (numpy.array(im1)[...,:3]/255)).astype('uint8'))

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

def spec_size(size):
    if isinstance(size, int): dims = (size, size)
    if isinstance(size, torch.Tensor): size = size.shape[:2]
    if isinstance(size, PIL.Image.Image): size = (size.size[1], size.size[0])
    if size is None: size = (224, 224)
    return size



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

    print("key: ", model.state_dict()["fc2.weight"])

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
        
    print("obs: ", observation.shape)
    size = 224
    layer = 'conv2'
    unit_num = 0 
    observation = frames[4]
    with Trace(model, layer) as tr:
        state = torch.tensor([observation],dtype=torch.float).to('cuda')
        preds = model(state)
        overlay_pic = tr.output[0, unit_num].detach().cpu()
        print("overlay : ", overlay_pic.shape)
    #     # print(np.array(overlay_pic).shape)
    #     print("tr.output: ",tr.output.max(3)[0].max(2)[0].topk(k=64)[1][0])
    #     plt.imshow(overlay_pic)
    #     plt.show()
    # with Trace(model, layer) as tr:
    #     im = np.einsum( 'ijk->jki', observation[:3])
    #     im = (im*255).astype(np.uint8)
    #     im = resize_and_crop(Image.fromarray(im), size)
    #     state = torch.tensor([observation],dtype=torch.float).to('cuda')
    #     preds = model(state)
    #     for unit in tr.output.max(3)[0].max(2)[0].topk(k=64)[1][0]:
    #         overlay_pic = overlay(im, rgb_heatmap(tr.output[0, unit_num].detach().cpu(), size = 224))

# def dissect_unit(ds, i, net, layer, unit):
#     data = ds[i]
#     im = np.einsum( 'ijk->jki', data[1:])
#     im = (im*255).astype(np.uint8)
#     im = resize_and_crop(Image.fromarray(im), size)
#     state = torch.tensor([data],dtype=torch.float).to('cuda')
#     with Trace(net, layer) as tr:
#         net(state)
#     mask = rgb_threshold(tr.output[0, unit].detach().cpu(), size=224)
# #     img = renormalize.as_image(data, source=ds)
#     return overlay_threshold(im, mask)

# ###################are these neuropns looking at the same thing#######################
# # neuron = 9
# # layer = 'conv1'
# # scores = []
# # for imagenum, observation in enumerate(pbar(frames)):
# #     with Trace(model, layer) as tr:
# #         im = np.einsum( 'ijk->jki', observation[:3])
# #         im = (im*255).astype(np.uint8)
# #         im = resize_and_crop(Image.fromarray(im), size)
# #         state = torch.tensor([observation],dtype=torch.float).to('cuda')
# #         preds = model(state)
# #     score = tr.output[0, neuron].view(-1).max()
# #     scores.append((score, imagenum))
# # scores.sort(reverse=True)

# # # show(f'{layer} neuron {neuron}',
# # #      [[dissect_unit(frames, scores[i][1], model, layer, neuron) for i in range(12)]])

# # image = dissect_unit(frames, scores[0][1], model, layer, neuron) 
# # plt.imshow(image)
# # plt.show()

# #### smooth grad cam###########
# size = 224

# for observation in frames:
# # observation = frames[4]
#     with torch.no_grad():
#         im = np.einsum( 'ijk->jki', observation[:3])
#         im = (im*255).astype(np.uint8)
#         im = resize_and_crop(Image.fromarray(im), size)
#         state = torch.tensor([observation],dtype=torch.float)
#         preds = model(state.to('cuda'))
#         # print(f'ACTION: {preds.argmax().item()}')
#         label = preds.argmax().item()
#     total = 0
#     for i in range(30):
#         im = np.einsum( 'ijk->jki', observation[:3])
#         im = (im*255).astype(np.uint8)
#         im = resize_and_crop(Image.fromarray(im), size)
#         state = torch.tensor([observation],dtype=torch.float)

#         prober = state + torch.randn(state.shape) 
#         prober.requires_grad = True
#         loss = torch.nn.functional.cross_entropy(
#             model(prober.to('cuda')),
#             torch.tensor([label]).to('cuda'))
#         loss.backward()

#         gradient = prober.grad # TO-DO (Replace None with the gradient wrt to the perturbed input)

#         total += gradient**2
#         prober.grad = None
#     camoutput = renormalize.as_image((total / total.max() * 5).clamp(0, 1)[0][1:], source='pt')

#     camoutput = resize_and_crop(camoutput, size)
#     #     im
#     #     camoutput
#     #     overlay(im, camoutput)
#     del prober