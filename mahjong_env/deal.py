from mahjong_env.feature import Suit, Tile
import json
from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.feature import FeatureAgent
from mahjong_env.tile import Wind, Tile, Suit
from mahjong_env.util import convert_string_to_tile, convert_string_to_wall
import numpy as np

def _append_observation(data, observation):
    """同一次重放同时保存旧/新特征，保证 A/B 数据逐样本对齐。"""
    data['state']['observation'].append(observation['observation'])
    data['state']['rich_observation'].append(observation['rich_observation'])
    data['state']['match_observation'].append(observation['match_observation'])
    data['state']['action_mask'].append(observation['action_mask'])


def deal(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mahjongsouldata = data["data"]["data"]["actions"]
    game = ThreePlayerMahjong()
    agent = FeatureAgent(Wind.East, 1)
    f = agent.response2action
    round_num = 0
    is_operation = True # Play/BuGang/AnGang 后其他人有没有操作(Peng/Gang/Hu)
    is_babei_operation = True # Pei 后其他人有没有操作(Hu)
    
    obs = [[], [], [], []]
    data = {
        'state': {
            'observation': [],
            'rich_observation': [],
            'match_observation': [],
            'action_mask': []
            },
        'label': [],
    }
    
    for action in mahjongsouldata:
        # try:
            if action["type"] == 1:
                result = action["result"]
                result_data = result["data"]
                name = result["name"]
                if "RecordNewRound" in name:
                    paishan = result_data["paishan"]
                    # print("Round", round_num)
                    round_num += 1
                    config = {}
                    wind_order = [Wind.East, Wind.South, Wind.West]
                    config["prevailing_wind"] = wind_order[result_data["chang"]]
                    config["round_number"] = result_data["ju"]
                    config["honba"] = result_data["ben"]
                    config["riichi_sticks"] = result_data["liqibang"]
                    config["scores"] = result_data.get(
                        "scores", [35000, 35000, 35000]
                    )
                    config["round_index"] = result_data["chang"] * 3 + result_data["ju"]
                    config["total_rounds"] = 6
                    operation = result_data["operation"]
                    current_player = operation["seat"]
                    wall = convert_string_to_wall(paishan)
                    obs = game.reset(wall, current_player, config)
                    is_operation = True
                    is_babei_operation = True
                elif "RecordDiscardTile" in name:
                    seat = result_data["seat"]
                    tile = result_data["tile"]
                    is_liqi = result_data["is_liqi"]
                    if is_liqi:
                        obs = game._obs()
                        _append_observation(data, obs[game.agent_names[seat]])
                        labels = np.zeros(agent.ACT_SIZE)
                        labels[agent.OFFSET_ACT['Riichi'] + agent._tile_to_index(convert_string_to_tile(tile))] = 1
                        data['label'].append(labels)
                        game.step({f"player_{seat}" : f(f"Riichi {convert_string_to_tile(tile)}")})
                    else:
                        obs = game._obs()
                        _append_observation(data, obs[game.agent_names[seat]])
                        labels = np.zeros(agent.ACT_SIZE)
                        labels[agent.OFFSET_ACT['Play'] + agent._tile_to_index(convert_string_to_tile(tile))] = 1
                        data['label'].append(labels)
                        game.step({f"player_{seat}" : f(f"Play {convert_string_to_tile(tile)}")})
                    if "operations" in result_data:
                        is_operation = False
                    else:
                        actions = {f"player_{(i + seat) % 3}" : f("Pass") for i in range(1, 3)}
                        game.step(actions)
                elif "RecordChiPengGang" in name:
                    seat = result_data["seat"]
                    data_type = result_data["type"]
                    if data_type == 1:
                        current_player = game.current_player
                        is_operation = True
                        tiles = result_data["tiles"]
                        froms = result_data["froms"]
                        own_tiles = [
                            code for code, source in zip(tiles, froms)
                            if source == seat
                        ]
                        if not own_tiles:
                            raise ValueError("碰牌记录中没有玩家自己的牌")
                        # 动作编号要区分“用赤五碰”和“用普通五碰”。
                        tile = next(
                            (code for code in own_tiles if code in {"0p", "0s"}),
                            own_tiles[0],
                        )
                        obs = game._obs()
                        _append_observation(data, obs[game.agent_names[seat]])
                        labels = np.zeros(agent.ACT_SIZE)
                        labels[agent.OFFSET_ACT['Peng'] + agent._tile_to_index(convert_string_to_tile(tile))] = 1
                        data['label'].append(labels)
                        actions = {f"player_{seat}": f(f"Peng {convert_string_to_tile(tile)}")}
                        for i in range(3):
                            if i != current_player and i != seat:
                                actions[f"player_{i}"] = f("Pass")
                        game.step(actions)
                    if data_type == 2:
                        current_player = game.current_player
                        is_operation = True
                        called = game.curTile
                        tile = Tile(called.suit, called.value, is_red=False)
                        obs = game._obs()
                        _append_observation(data, obs[game.agent_names[seat]])
                        labels = np.zeros(agent.ACT_SIZE)
                        labels[agent.OFFSET_ACT['Kang'] + agent._tile_to_index(tile)] = 1
                        data['label'].append(labels)
                        actions = {f"player_{seat}": f(f"Kang {tile}")}
                        for i in range(3):
                            if i != current_player and i != seat:
                                actions[f"player_{i}"] = f("Pass")
                        game.step(actions)
                elif "RecordAnGangAddGang" in name:
                    seat = result_data["seat"]
                    data_type = result_data["type"]
                    tile = result_data["tiles"]
                    if data_type == 2:
                        obs = game._obs()
                        _append_observation(data, obs[game.agent_names[seat]])
                        labels = np.zeros(agent.ACT_SIZE)
                        labels[agent.OFFSET_ACT['BuKang'] + agent._tile_to_index(convert_string_to_tile(tile))] = 1
                        data['label'].append(labels)
                        actions = {f"player_{seat}": f(f"BuKang {convert_string_to_tile(tile)}")}
                    elif data_type == 3:
                        obs = game._obs()
                        _append_observation(data, obs[game.agent_names[seat]])
                        labels = np.zeros(agent.ACT_SIZE)
                        labels[agent.OFFSET_ACT['AnKang'] + agent._tile_to_index(convert_string_to_tile(tile))] = 1
                        data['label'].append(labels)
                        actions = {f"player_{seat}": f(f"AnKang {convert_string_to_tile(tile)}")}
                    game.step(actions)
                    if "operations" in result_data:
                        pass
                    else:
                        actions = {f"player_{(i + seat) % 3}" : f("Pass") for i in range(1, 3)}
                        game.step(actions)
                elif "RecordDealTile" in name:
                    seat = result_data["seat"]
                    if not is_babei_operation:
                        is_babei_operation = True
                        obs = game._obs()
                        for i in range(1, 3):
                            _append_observation(data, obs[game.agent_names[(seat + i) % 3]])
                            labels = np.zeros(agent.ACT_SIZE)
                            labels[agent.OFFSET_ACT['Pass']] = 1
                            data['label'].append(labels)
                        game.step({f"player_{(seat + i) % 3}": f("Pass") for i in range(1, 3)})
                    if not is_operation:
                        is_operation = True
                        obs = game._obs()
                        for i in range(2):
                            _append_observation(data, obs[game.agent_names[(seat + i) % 3]])
                            labels = np.zeros(agent.ACT_SIZE)
                            labels[agent.OFFSET_ACT['Pass']] = 1
                            data['label'].append(labels)
                        game.step({f"player_{(seat + i) % 3}": f("Pass") for i in range(2)})
                elif "RecordHule" in name:
                    actions = {}
                    is_babei_operation = True
                    for hules in result_data["hules"]:
                        seat = hules["seat"]
                        zimo = hules["zimo"]
                        fans = hules["fans"]
                        yiman = hules["yiman"]
                        fu = hules["fu"]
                        obs = game._obs()
                        _append_observation(data, obs[game.agent_names[seat]])
                        labels = np.zeros(agent.ACT_SIZE)
                        labels[agent.OFFSET_ACT['Hu']] = 1
                        data['label'].append(labels)
                        actions[f"player_{seat}"] = f("Hu")
                        val = 0
                        for fan in fans:
                            val += fan["val"]
                        if yiman:
                            val = -val
                        game.set_fan_fu(seat, val, fu)
                    if zimo:
                        game.step(actions)
                    else:
                        for i in range(3):
                            if i == game.current_player or f"player_{i}" in actions:
                                continue
                            actions[f"player_{i}"] = f("Pass")
                        game.step(actions)
                elif "RecordBaBei" in name:
                    seat = result_data["seat"]
                    obs = game._obs()
                    _append_observation(data, obs[game.agent_names[seat]])
                    labels = np.zeros(agent.ACT_SIZE)
                    labels[agent.OFFSET_ACT['Pei']] = 1
                    data['label'].append(labels)
                    game.step({f"player_{seat}" : f("Pei")})
                    if "operations" in result_data:
                        is_babei_operation = False
                    else:
                        actions = {f"player_{(i + seat) % 3}" : f("Pass") for i in range(1, 3)}
                        game.step(actions)

                        
            if action["type"] == 2:
                user_input = action["user_input"]
                seat = user_input["seat"]
                user_type = user_input["type"]
                if user_type == 2:
                    operation = user_input["operation"]
                    if "cancel_operation" in operation:
                        continue
                    op_type = operation["type"]
    
    return data
                    # if op_type == 11:
                    #     game.step({f"player_{seat}" : f("Pei")})
            # if action["type"] == 3:
            #     user_event = action["user_event"]
            #     seat = user_event["seat"]
            #     event_type = user_event["type"]
            #     if event_type == 2:
            #         raise Exception("断线")
        # except Exception as e:
        #     print(action)
        #     print(e)
        #     break
