from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.feature import FeatureAgent
from mahjong_env.tile import Wind
from model import CNN
import random
import torch

def get_random_one_index(lst):
    for i in range(30, 176):
        if lst[i] == 1:
            return i
    
    indices = [i for i, x in enumerate(lst) if x == 1]  # 找出所有1的索引
    return random.choice(indices) if indices else None  # 随机选一个，没有1则返回None

if __name__ == "__main__":
    game = ThreePlayerMahjong()
    obs = game.reset()
    done = False
    agent = FeatureAgent(Wind.East, 1)
    model = CNN()
    model.load_state_dict(torch.load('./model/model.pt'))
    while not done:
        actions = {}
        for agent_name in obs:
            state = obs[agent_name]
            state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
            state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
            model.train(False)
            with torch.no_grad():
                logits = model(state)
                action_dist = torch.distributions.Categorical(logits = logits)
                action = action_dist.sample().item()
            actions[agent_name] = action
        try:
            next_obs, rewards, done = game.step(actions)
        except Exception as e:
            print(e)
            break
        obs = next_obs
    # print(obs)
    print("finish")
