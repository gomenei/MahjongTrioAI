from mahjong_env.feature import Suit, Tile
from typing import List
import re
import json
from mahjong_env.game import ThreePlayerMahjong
from mahjong_env.feature import FeatureAgent, Tile, Suit

def convert_string_to_wall(tile_str: str) -> List[Tile]:
    """将字符串格式的牌转换为牌山列表"""
    wall = []
    pattern = re.compile(r'([0-9]+[mpsz])')  # 匹配数字+花色的组合（如8p、1z）
    tokens = pattern.findall(tile_str)  # 分割为['8p','1z','4p','8p',...]
    
    for token in tokens[::-1]:  # 逆序解析（假设牌山从末尾取牌）
        value = int(token[:-1])
        suit_char = token[-1].lower()
        
        # 确定花色
        suit_map = {
            'm': Suit.Manzu,
            'p': Suit.Pinzu,
            's': Suit.Souzu,
            'z': Suit.Honors
        }
        suit = suit_map.get(suit_char)
        if not suit:
            raise ValueError(f"无效的花色字符: {suit_char}")
        
        # 处理红宝牌（红5标记为0，如0p=红5筒）
        is_red = (value == 0) and (suit != Suit.Honors)  # 字牌无红宝牌
        actual_value = 5 if value == 0 else value  # 0转换为5
        
        wall.append(Tile(suit, actual_value, is_red))
    return wall

def convert_string_to_tile(tile_str: str) -> Tile:
    """
    将单张麻将牌字符串转换为 Tile 对象
    格式示例: 
        - "5m" → 5万 
        - "0p" → 红5筒 (is_red=True)
        - "7z" → 7字牌（东风）
    
    Args:
        tile_str: 输入的牌字符串（如 "5m"）
    
    Returns:
        Tile: 对应的麻将牌对象
    
    Raises:
        ValueError: 输入格式无效时抛出异常
    """
    if not tile_str or len(tile_str) < 2:
        raise ValueError(f"无效的牌字符串格式: {tile_str}")

    # 解析数值和花色
    value_part = tile_str[:-1]
    suit_char = tile_str[-1].lower()

    try:
        value = int(value_part)
    except ValueError:
        raise ValueError(f"牌字符串的数值部分必须是数字: {tile_str}")

    # 确定花色
    suit_map = {
        'm': Suit.Manzu,   # 万
        'p': Suit.Pinzu,   # 筒
        's': Suit.Souzu,   # 条
        'z': Suit.Honors   # 字牌
    }
    suit = suit_map.get(suit_char)
    if not suit:
        raise ValueError(f"无效的花色字符: {suit_char}")

    # 处理红宝牌（红5标记为0，如 "0p"=红5筒）
    is_red = (value == 0) and (suit != Suit.Honors)  # 字牌无红宝牌
    actual_value = 5 if (value == 0 and suit != Suit.Honors) else value

    # 检查数值范围
    if suit == Suit.Honors:
        if not (1 <= actual_value <= 7):
            raise ValueError(f"字牌数值必须在1-7之间: {tile_str}")
    else:
        if not (1 <= actual_value <= 9):
            raise ValueError(f"数牌数值必须在1-9之间: {tile_str}")

    return Tile(suit, actual_value, is_red)

def deal(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data["data"]["data"]["actions"]
    game = ThreePlayerMahjong()
    agent = FeatureAgent(Tile(Suit.Honors, 1))
    f = agent.response2action
    round_num = 0
    is_operation = True # Play/BuGang/AnGang 后其他人有没有操作(Peng/Gang/Hu)
    for action in data:
        # try:
            if action["type"] == 1:
                result = action["result"]
                result_data = result["data"]
                name = result["name"]
                if "RecordNewRound" in name:
                    paishan = result_data["paishan"]
                    # print("Round", round_num)
                    round_num += 1
                    operation = result_data["operation"]
                    current_player = operation["seat"]
                    wall = convert_string_to_wall(paishan)
                    game.reset(wall, current_player)
                elif "RecordDiscardTile" in name:
                    seat = result_data["seat"]
                    tile = result_data["tile"]
                    is_liqi = result_data["is_liqi"]
                    if is_liqi:
                        game.step({f"player_{seat}" : f(f"Riichi {convert_string_to_tile(tile)}")})
                    else:
                        game.step({f"player_{seat}" : f(f"Play {convert_string_to_tile(tile)}")})
                    if "operations" in result_data:
                        is_operation = False
                    else:
                        actions = {f"player_{(i + seat) % 3}" : f("Pass") for i in range(1, 3)}
                        game.step(actions)
                elif "RecordBaBei" in name:
                    seat = result_data["seat"]
                    actions = {f"player_{(i + seat) % 3}" : f("Pass") for i in range(1, 3)}
                    game.step(actions)
                elif "RecordChiPengGang" in name:
                    seat = result_data["seat"]
                    data_type = result_data["type"]
                    if data_type == 1:
                        current_player = game.current_player
                        is_operation = True
                        actions = {f"player_{seat}": f(f"Peng {convert_string_to_tile(tile)}")}
                        for i in range(3):
                            if i != current_player and i != seat:
                                actions[f"player_{i}"] = f("Pass")
                        game.step(actions)
                    if data_type == 2:
                        current_player = game.current_player
                        is_operation = True
                        actions = {f"player_{seat}": f(f"Kang {convert_string_to_tile(tile)}")}
                        for i in range(3):
                            if i != current_player and i != seat:
                                actions[f"player_{i}"] = f("Pass")
                        game.step(actions)
                elif "RecordAnGangAddGang" in name:
                    seat = result_data["seat"]
                    data_type = result_data["type"]
                    tile = result_data["tiles"]
                    if data_type == 2:
                        actions = {f"player_{seat}": f(f"BuKang {convert_string_to_tile(tile)}")}
                    elif data_type == 3:
                        actions = {f"player_{seat}": f(f"AnKang {convert_string_to_tile(tile)}")}
                    game.step(actions)
                    if "operations" in result_data:
                        pass
                    else:
                        actions = {f"player_{(i + seat) % 3}" : f("Pass") for i in range(1, 3)}
                        game.step(actions)
                elif "RecordDealTile" in name:
                    seat = result_data["seat"]
                    if not is_operation:
                        is_operation = True
                        game.step({f"player_{(seat + i) % 3}": f("Pass") for i in range(2)})

                        
            if action["type"] == 2:
                user_input = action["user_input"]
                seat = user_input["seat"]
                user_type = user_input["type"]
                if user_type == 2:
                    operation = user_input["operation"]
                    if "cancel_operation" in operation:
                        continue
                    op_type = operation["type"]
                    if op_type == 11:
                        game.step({f"player_{seat}" : f("Pei")})
            # if action["type"] == 3:
            #     user_event = action["user_event"]
            #     seat = user_event["seat"]
            #     event_type = user_event["type"]
            #     if event_type == 2:
            #         raise Exception("断线")
