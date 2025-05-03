import json
import collections
import os
import math
from typing import List, Dict, Tuple, Optional, Any, Counter as TypingCounter

# --- Tile Parsing and Utility Function ---
def parse_tiles(tile_str: str) -> List[str]:
    """解析输入牌型字符串'123m45p789s11z' 为列表， 红宝牌转换为5
    ['1m', '2m', '3m', '4p', '5p', '7s', '8s', '9s', '1z', '1z']"""
    tiles = []
    i = 0
    while i < len(tile_str):
        num_str = ""
        while i < len(tile_str) and tile_str[i].isdigit():
            num_str += tile_str[i]
            i += 1
        if i < len(tile_str):
            suit = tile_str[i]
            i += 1
            for num_char in num_str:
                if num_char == '0':  # 红宝牌处理
                    tiles.append(f"5{suit}")  # 转换为5对应牌型
                else:
                    tiles.append(f"{num_char}{suit}")
        else:
            break
    return tiles

def get_tile_counts(tiles: List[str]) -> collections.Counter:
    #统计列表中各牌的出现次数
    return collections.Counter(tiles)

def get_tile_key(num: int, suit: str) -> str:
    """Creates the standard string representation from number and suit."""
    return f"{num}{suit}"


# --- Core Logic ---
PATTERNS_3N = None
PATTERNS_3NP2 = None
PATTERNS_shuntsu = None
PATTERNS_shuntsup2 = None
PATTERNS_liangbeikou = None
PATTERNS_liangbeikoup2 = None
PATTERNS_LOADED = False
PATTERN_LOAD_ERROR = False

def load_patterns():
    """Loads the JSON pattern files into global variables."""
    global PATTERNS_3N, PATTERNS_3NP2, PATTERNS_shuntsu,PATTERNS_shuntsup2
    global PATTERNS_liangbeikou,PATTERNS_liangbeikoup2
    global PATTERNS_LOADED, PATTERN_LOAD_ERROR

    if PATTERNS_LOADED or PATTERN_LOAD_ERROR:
        return not PATTERN_LOAD_ERROR

    script_dir = os.path.dirname(__file__) # Get directory of the current script
    path_3n = os.path.join(script_dir, '3n_patterns.json')
    path_3np2 = os.path.join(script_dir, '3np2_patterns.json')
    path_shun = os.path.join(script_dir, 'shuntsu_patterns.json')
    path_shunp2 = os.path.join(script_dir, 'shuntsup2_patterns.json')
    path_liangbeikou = os.path.join(script_dir, 'liangbeikou.json')
    path_liangbeikoup2 = os.path.join(script_dir, 'liangbeikoup2.json')
    try:
        with open(path_3n, 'r', encoding='utf-8') as f:
            PATTERNS_3N = json.load(f)
        with open(path_3np2, 'r', encoding='utf-8') as f:
            PATTERNS_3NP2 = json.load(f)
        with open(path_shun, 'r', encoding='utf-8') as f:
            PATTERNS_shuntsu = json.load(f)
        with open(path_shunp2, 'r', encoding='utf-8') as f:
            PATTERNS_shuntsup2 = json.load(f)
        with open(path_liangbeikou, 'r', encoding='utf-8') as f:
            PATTERNS_liangbeikou = json.load(f)
        with open(path_liangbeikoup2, 'r', encoding='utf-8') as f:
            PATTERNS_liangbeikoup2 = json.load(f)
        PATTERNS_LOADED = True
        return True
    except FileNotFoundError:
        print(f"Error: Pattern files not found. Looked for '{path_3n}' and '{path_3np2}'.")
        PATTERN_LOAD_ERROR = True
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from pattern files. Details: {e}")
        PATTERN_LOAD_ERROR = True
        return False
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading pattern files: {e}")
        PATTERN_LOAD_ERROR = True
        return False

def check_standard_hand_3n(counts: TypingCounter[str], n3_patterns) -> bool:
    """检查给定的牌（用计数表示）是否可以形成标准手牌,3*N+2"""
    global PATTERNS_LOADED, PATTERN_LOAD_ERROR
    if not PATTERNS_LOADED:
      raise RuntimeError("Failed to load necessary pattern files for hand evaluation.")
    total_tiles = sum(counts.values())
    if total_tiles % 3 != 0:
        return False
    if any(count > 4 for count in counts.values()):
        return False
    # --- 1. Separate Honors and Numbers, Process Honors ---
    number_components_raw = {'m': {}, 'p': {}, 's': {}}
    num_honor_triplets = 0
    current_counts = counts.copy() # Work with a copy
    honor_keys_to_remove = []
    for tile, count in current_counts.items():
        num, suit = tile[0],tile[1]
        if suit == 'z':
            honor_keys_to_remove.append(tile)
            if count == 3:
                num_honor_triplets += 1
            else:
                return False
        elif suit in ['m', 'p', 's']:
            if count > 0: # Only add tiles present
                number_components_raw[suit][num] = count
    # --- 3. Group Number Tiles into Connected Components ---
    number_components = [] # List of components: {'suit': str, 'start': int, 'seq': str, 'count': int}
    for suit in ['m', 'p', 's']:
        if not number_components_raw[suit]:
            continue
        sorted_nums = sorted(number_components_raw[suit].keys())
        current_component_start = -1
        current_sequence = []
        current_tile_count = 0
        for i, num in enumerate(sorted_nums):
            count = number_components_raw[suit][num]
            if count > 4: return False # Redundant check, but safe
            if current_component_start == -1: # Start of a new component
                current_component_start = num
                current_sequence.append(str(count))
                current_tile_count += count
            elif int(num) == int(sorted_nums[i-1]) + 1: # Continues the current component
                current_sequence.append(str(count))
                current_tile_count += count
            else: # Gap found, end previous component and start new one
                # Store previous component
                number_components.append({
                    'suit': suit,
                    'start': current_component_start,
                    'seq': "".join(current_sequence),
                    'count': current_tile_count
                })
                # Start new component
                current_component_start = num
                current_sequence = [str(count)]
                current_tile_count = count
        # Store the last component for the suit
        if current_component_start != -1:
            number_components.append({
                'suit': suit,
                'start': current_component_start,
                'seq': "".join(current_sequence),
                'count': current_tile_count
            })
    
    # --- 4. Validate Component Sizes ---
    total_number_component_tiles = 0
    for comp in number_components:
        comp_count = comp['count']
        total_number_component_tiles += comp_count
        if comp_count % 3 != 0:
             return False
    
    # Check if total tiles match accounted tiles
    if total_number_component_tiles + 3* num_honor_triplets != total_tiles:
         print(f"DEBUG: Tile count mismatch. Total={total_tiles}, AccountedHonor={3* num_honor_triplets}, AccountedNumber={total_number_component_tiles}")
         return False # Should not happen if logic is correct

    # --- 5. Check Topological Sequences against Patterns ---
    for comp in number_components:
        comp_count = comp['count']
        comp_seq = comp['seq']
        n = comp_count // 3
        n_key = str(n)
        if n < 0: return False # Invalid component size resulted in N<0
        if n_key not in n3_patterns:
            # The required number of melds (N) doesn't exist in the patterns file
            # print(f"DEBUG: N={n_key} not found in {'3n'} patterns for component {comp_seq}")
            return False
        valid_sequences_for_n = n3_patterns[n_key]
        if comp_seq not in valid_sequences_for_n:
            return False
    # If all checks passed
    return {"type": "standard", "number_components": number_components}

def check_standard_hand(counts: TypingCounter[str], n3_patterns, n3p2_patterns) -> bool:
    """检查给定的牌（用计数表示）是否可以形成标准手牌,3*N+2"""
    global PATTERNS_LOADED, PATTERN_LOAD_ERROR
    if not PATTERNS_LOADED:
      raise RuntimeError("Failed to load necessary pattern files for hand evaluation.")
    total_tiles = sum(counts.values())
    if total_tiles % 3 != 2:
        return False
    if any(count > 4 for count in counts.values()):
        return False
    # --- 1. Separate Honors and Numbers, Process Honors ---
    number_components_raw = {'m': {}, 'p': {}, 's': {}}
    num_honor_triplets = 0
    num_honor_pairs = 0
    honor_tiles_accounted = 0
    current_counts = counts.copy() # Work with a copy
    honor_keys_to_remove = []
    for tile, count in current_counts.items():
        num, suit = tile[0],tile[1]
        if suit == 'z':
            honor_keys_to_remove.append(tile)
            if count == 1 or count > 4: # Cannot be decomposed
                return False
            elif count == 2:
                num_honor_pairs += 1
                honor_tiles_accounted += 2
            elif count == 3:
                num_honor_triplets += 1
                honor_tiles_accounted += 3
            elif count == 4:
                return False
        elif suit in ['m', 'p', 's']:
            if count > 0: # Only add tiles present
                number_components_raw[suit][num] = count
    if num_honor_pairs > 1:
        return False
    # print(number_components_raw,num_honor_triplets,num_honor_pairs)
    # --- 2. Check Overall Structure Feasibility based on Honors ---
    needs_pair = (num_honor_pairs == 0)

    # --- 3. Group Number Tiles into Connected Components ---
    number_components = [] # List of components: {'suit': str, 'start': int, 'seq': str, 'count': int}
    for suit in ['m', 'p', 's']:
        if not number_components_raw[suit]:
            continue
        sorted_nums = sorted(number_components_raw[suit].keys())
        current_component_start = -1
        current_sequence = []
        current_tile_count = 0
        for i, num in enumerate(sorted_nums):
            count = number_components_raw[suit][num]
            if count > 4: return False # Redundant check, but safe
            if current_component_start == -1: # Start of a new component
                current_component_start = num
                current_sequence.append(str(count))
                current_tile_count += count
            elif int(num) == int(sorted_nums[i-1]) + 1: # Continues the current component
                current_sequence.append(str(count))
                current_tile_count += count
            else: # Gap found, end previous component and start new one
                # Store previous component
                number_components.append({
                    'suit': suit,
                    'start': current_component_start,
                    'seq': "".join(current_sequence),
                    'count': current_tile_count
                })
                # Start new component
                current_component_start = num
                current_sequence = [str(count)]
                current_tile_count = count

        # Store the last component for the suit
        if current_component_start != -1:
            number_components.append({
                'suit': suit,
                'start': current_component_start,
                'seq': "".join(current_sequence),
                'count': current_tile_count
            })
    #print(number_components)
    # --- 4. Validate Component Sizes ---
    num_components_need_pair = 0
    total_number_component_tiles = 0
    for comp in number_components:
        comp_count = comp['count']
        total_number_component_tiles += comp_count
        if comp_count % 3 == 1:
             return False
        elif comp_count % 3 == 2:
             num_components_need_pair += 1

    # Check if total tiles match accounted tiles
    if total_number_component_tiles + honor_tiles_accounted != total_tiles:
         print(f"DEBUG: Tile count mismatch. Total={total_tiles}, AccountedHonor={honor_tiles_accounted}, AccountedNumber={total_number_component_tiles}")
         return False # Should not happen if logic is correct

    # Check if the pair structure matches
    if needs_pair:
        if num_components_need_pair != 1:
            return False # Exactly one number component must provide the pair
    else: # Pair provided by honors
        if num_components_need_pair != 0:
            return False # All number components must be 3N

    # --- 5. Check Topological Sequences against Patterns ---
    for comp in number_components:
        comp_count = comp['count']
        comp_seq = comp['seq']
        needs_comp_pair = (comp_count % 3 == 2)

        if comp_count == 0 : continue # Should not happen, but safe check

        patterns_to_use = n3p2_patterns if needs_comp_pair else n3_patterns
        n = (comp_count - 2) // 3 if needs_comp_pair else comp_count // 3
        n_key = str(n)

        if n < 0: return False # Invalid component size resulted in N<0

        if n_key not in patterns_to_use:
            # The required number of melds (N) doesn't exist in the patterns file
            #print(f"DEBUG: N={n_key} not found in {'3np2' if needs_comp_pair else '3n'} patterns for component {comp_seq}")
            return False

        valid_sequences_for_n = patterns_to_use[n_key]
        if comp_seq not in valid_sequences_for_n:
            return False
    # If all checks passed
    return {"type": "standard", "number_components": number_components}


def check_special_hands(counts: collections.Counter) -> Optional[Dict[str, Any]]:
    """Checks for special hand patterns like Seven Pairs and Thirteen Orphans."""
    num_tiles = sum(counts.values())
    if num_tiles != 14: # Special hands require exactly 14 tiles
        return False

    # 1. Check for Seven Pairs (Chiitoitsu)
    if len(counts) == 7 and all(count == 2 for count in counts.values()):
        # Needs exact check for 7 pairs
        pairs_list = sorted(list(counts.keys())) # List of the tile types that form pairs
        return {"type": "chiitoitsu", "pairs": pairs_list}

    # 2. Check for Thirteen Orphans (Kokushi Musou)
    # Requires one of each terminal (1,9 mps) and honor (1-7z), plus one duplicate.
    terminals_and_honors = {
        "1m", "9m", "1p", "9p", "1s", "9s",
        "1z", "2z", "3z", "4z", "5z", "6z", "7z" # E,S,W,N,Wh,G,R
    }
    is_kokushi = True
    duplicate_tile = None
    if len(counts) != 13: # Must have exactly 13 unique tiles for Kokushi
        is_kokushi = False
    else:
        for tile in terminals_and_honors:
            count = counts.get(tile, 0)
            if count == 0:
                is_kokushi = False
                break
            if count == 2:
                if duplicate_tile is not None: # More than one duplicate
                    is_kokushi = False
                    break
                duplicate_tile = tile
            elif count != 1: # Must be 1 or 2
                is_kokushi = False
                break
        if duplicate_tile is None: # Must have exactly one duplicate tile
             is_kokushi = False

    if is_kokushi and duplicate_tile is not None:
        return {"type": "kokushi", "wait_tile": duplicate_tile} # Or indicate 13-way wait if counts are all 1

    return False

# --- Score Calculation Logic ---

def calculate_fu(decomposition: Dict[str, Any], context: Dict[str, Any], outer_melds: List[Dict],all_tiles: List[str], pinhe: bool) -> int:
    if decomposition:
        win_type = decomposition.get("type", "unknown")
        if win_type == "chiitoitsu":return 25
        if win_type == "kokushi":return 20
        # 1 底符
        fu = 20
        # 2 手牌
        yaojiu = {"1m","9m","1s","9s","1p","9p","1z","2z","3z","4z","5z", "6z", "7z"}
        all_tiles_counts = get_tile_counts(all_tiles)
        open_minkou = 0
        for meld in outer_melds:
            if meld.get("type") == "minkou":
                open_minkou += (1 + (meld.get("tiles")[0] in yaojiu))
        def cal_closed_ankou(counts):
            suits = {'m','p','s','z'}
            depth = [0]
            for suit in suits:
                for num in range(1,10-2*(suit=='z')):
                    t = f"{num}{suit}"
                    if counts.get(t,0) >= 3:
                        temp_counts = counts.copy()
                        temp_counts[t] -= 3
                        if temp_counts[t] == 0:del temp_counts[t]
                        if check_standard_hand(temp_counts,PATTERNS_3N,PATTERNS_3NP2):
                            depth.append(cal_closed_ankou(temp_counts)+ 1 + (t in yaojiu)) 
            return max(depth)
        closed_ankou = cal_closed_ankou(all_tiles_counts)
        open_minkan = 0
        for meld in outer_melds:
            if meld.get("type") == "minkan":
                open_minkan += (1 + (meld.get("tiles")[0] in yaojiu))
        open_ankan = 0
        for meld in outer_melds:
            if meld.get("type") == "ankan":
                open_minkan += (1 + (meld.get("tiles")[0] in yaojiu))
        mianzi = 2* open_minkou + 4 * closed_ankou + 8* open_minkan + 16* open_ankan
        fu += mianzi
        jantou = 0
        if context['selfwind'] == context['placewind']:
            t = f"{context['selfwind']+1}z"
            if all_tiles_counts.get(t,0) == 2:jantou = 4
        else:
            for t in {f"{context['selfwind']+1}z",f"{context['placewind']+1}z","5z","6z","7z"}:
                if all_tiles_counts.get(t,0) == 2:jantou = 2
        fu += jantou
        # 3 听牌
        tinpai = 0
        jinzhang = parse_tiles(context.get("jinzhang", ""))[0]
        # 3.1 单骑听牌
        if all_tiles_counts.get(jinzhang,0) >= 2:
            temp_counts = all_tiles_counts.copy()
            temp_counts[jinzhang] -= 2
            if temp_counts[jinzhang] == 0:del temp_counts[jinzhang]
            if check_standard_hand_3n(temp_counts,PATTERNS_3N): tinpai =2
        # 3.2 边张听牌
        if "z" not in jinzhang and ("3" in jinzhang or "7" in jinzhang):
            suit = jinzhang[1]
            if "3" in jinzhang:
                seq_tiles = [f"1{suit}", f"2{suit}", f"3{suit}"]
            else:
                seq_tiles = [f"7{suit}", f"8{suit}", f"9{suit}"]
            if all(all_tiles_counts.get(t,0) >= 1 for t in seq_tiles):
                temp_counts = all_tiles_counts.copy()
                for t in seq_tiles:
                    temp_counts[t] -= 1
                    if temp_counts[t] == 0:
                        del temp_counts[t]
                if check_standard_hand(temp_counts,PATTERNS_3N,PATTERNS_3NP2):
                    tinpai = 2
        # 3.3 嵌张听牌
        if "z" not in jinzhang and "1" not in jinzhang and "9" not in jinzhang:
            suit = jinzhang[1]
            num = int(jinzhang[0])
            seq_tiles = [f"{num-1}{suit}", f"{num}{suit}", f"{num+1}{suit}"]
            if all(all_tiles_counts.get(t,0) >= 1 for t in seq_tiles):
                temp_counts = all_tiles_counts.copy()
                for t in seq_tiles:
                    temp_counts[t] -= 1
                    if temp_counts[t] == 0:
                        del temp_counts[t]
                if check_standard_hand(temp_counts,PATTERNS_3N,PATTERNS_3NP2):
                    tinpai = 2
        fu += tinpai
        #print(mianzi,jantou,tinpai,fu)
        # 4 和牌
        def is_menzen_clear(outer_melds):
            for meld in outer_melds:
                # 如果存在非暗杠的副露，则门前清失效
                if meld["type"] not in ["ankan"]:
                    return False
            return True
        is_menzen = is_menzen_clear(outer_melds) # 是否门清
        if (not pinhe) and context.get("isTsumo", False): fu += 2
        #print(fu)
        if is_menzen and (not context.get("isTsumo", True)): fu += 10
        if pinhe and is_menzen: fu = 20
        if not is_menzen and fu<30:fu=30
        return math.ceil(fu / 10) * 10
    else:
        return 0


def calculate_fan(decomposition: Dict[str, Any], 
                  context: Dict[str, Any], 
                  outer_melds: List[Dict], 
                  all_tiles: List[str]) -> Tuple[int, List[str]]:
    fan = 0
    yaku_list = []
    def is_menzen_clear(outer_melds):
    # 遍历所有副露
        for meld in outer_melds:
            # 如果存在非暗杠的副露，则门前清失效
            if meld["type"] not in ["ankan"]:
                return False
        return True
    is_menzen = is_menzen_clear(outer_melds) # 是否门清
    jinzhang = parse_tiles(context.get("jinzhang", ""))[0]
    alls_tiles = all_tiles +  [tile for meld in outer_melds for tile in meld.get("tiles", [])]
    all_tiles_counts = get_tile_counts(all_tiles)
    alls_tiles_counts = get_tile_counts(alls_tiles)

    yakuhai = {"5z", "6z", "7z"}  # 假设5z=白 6z=发 7z=中
    # 场风
    yakuhai.add(f"{context['placewind']+1}z")
    # 自风
    yakuhai.add(f"{context['selfwind']+1}z")

    def has_yakuhai() -> bool:
        """检查是否包含役牌刻子/雀头"""
        return any(alls_tiles_counts.get(t, 0) >= 3 for t in yakuhai)
    
    def is_tanyao() -> bool:
        """断幺九判断"""
        for tile in alls_tiles:
            num, suit = tile[0], tile[1]
            if suit == 'z' or num in ['1', '9']:
                return False
        return True
    
    def is_pinfu() -> bool:
        """平和判断"""
        if len(outer_melds) > 0: return False
        if decomposition.get("type") != "standard": return False
        # 雀头不能是役牌
        tile_set = set(alls_tiles)
        if yakuhai.isdisjoint(tile_set): return False
        if any(alls_tiles_counts.get(t, 0)>=3 for t in {"1z","2z","3z","4z","5z", "6z", "7z"}): return False
        # 所有面子必须是顺子
        closed_counts = all_tiles_counts
        if not check_standard_hand(closed_counts.copy(),PATTERNS_shuntsu, PATTERNS_shuntsup2):
            return False
        #两面听
        if not jinzhang or jinzhang[-1] == 'z':
            return False
        num, suit = int(jinzhang[0]), jinzhang[1]
        possible_shuntsu = []
        if num <= 7:
            possible_shuntsu.append([jinzhang, f"{num+1}{suit}", f"{num+2}{suit}"])
        if num >= 3:
            possible_shuntsu.append([f"{num-2}{suit}", f"{num-1}{suit}", jinzhang])
        def is_pinfu_ryanmen():
            for shuntsu in possible_shuntsu:
                temp_tiles = all_tiles.copy()
                try:
                    for t in shuntsu:
                        temp_tiles.remove(t)
                except ValueError:
                    continue
                temp_counts = get_tile_counts(temp_tiles)
                if check_standard_hand(temp_counts.copy(),PATTERNS_shuntsu, PATTERNS_shuntsup2):return True
            return False
        if not is_pinfu_ryanmen():
            return False
        return True

    win_type = decomposition.get("type", "unknown")

    # --- Placeholder Yaku Checks ---
    """Yakuman"""
    # 1. 国士无双
    if win_type == "kokushi":
        if decomposition.get("wait_tile") == jinzhang:
            fan += -2
            yaku_list.append("国士无双十三面")
        else:
            fan += -1
            yaku_list.append("国士无双")
    # 2. 天和
    if context.get("tianhe",False):
        fan += -1
        yaku_list.append("天和")
    # 3. 地和
    if context.get("dihe",False):
        fan += -1
        yaku_list.append("地和")
    # 4. 四暗刻
    def check_counts(counts):
        # 统计计数值为 2 和 3 的个数
        num_twos = list(counts.values()).count(2)
        num_threes = list(counts.values()).count(3)
        if num_twos == 1 and num_threes == len(counts) - 1:
             # 找到计数为 2 的牌
            for tile, count in counts.items():
                if count == 2:
                    return tile
        return False
    jantou = check_counts(all_tiles_counts)
    if is_menzen and jantou:
        if jinzhang == jantou:
            fan += -2
            yaku_list.append("四暗刻单骑")
        else:
            if context.get("isTsumo",False):
                fan += -1
                yaku_list.append("四暗刻")
    # 5. 九宝莲灯
    if len(outer_melds) == 0:
        suits = {t[1] for t in all_tiles}
        base_valid = False
        if len(suits) == 1 and 'z' not in suits:
            suit = suits.pop()
            num_counts = collections.defaultdict(int)
            for tile in all_tiles:
                num_counts[tile[0]] += 1
            base_valid = True
            for num in ['1', '9']:
                if num_counts.get(num, 0) < 3:
                    base_valid = False
            for num in ['2', '3', '4', '5', '6', '7', '8']:
                if num_counts.get(num, 0) < 1:
                    base_valid = False
            total_tiles = sum(num_counts.values())
            if total_tiles == 14 and base_valid:
                for num in ['1', '9']:
                    if num_counts[num] != 3:
                        dup = num+suit
                for num in ['2', '3', '4', '5', '6', '7', '8']:
                    if num_counts[num] != 1:
                        dup = num+suit
                if jinzhang == dup:
                    fan += -2
                    yaku_list.append("纯正九莲宝灯")
                else:
                    fan += -1
                    yaku_list.append("九莲宝灯")
    # 6. 大三元
    def is_daisangen(tile_counts: collections.Counter):
        sangen_tiles = {"5z", "6z", "7z"}
        for tile in sangen_tiles:
            if tile_counts.get(tile, 0) < 3:
                return False
        return True
    if is_daisangen(alls_tiles_counts):
        fan += -1
        yaku_list.append("大三元")
    # 7. 字一色
    if all(tile.endswith('z') for tile in alls_tiles_counts):
        fan += -1
        yaku_list.append("字一色")
    # 8. 绿一色
    green_tiles = {'2s','3s','4s','6s','8s','6z'}
    if all(tile in green_tiles for tile in alls_tiles_counts):
        fan += -1
        yaku_list.append("绿一色")
    # 9. 清老头
    laotou = {"1m","9m","1s","9s","1p","9p"}
    if decomposition.get("type") == "standard" and all(tile in laotou for tile in alls_tiles_counts):
        fan += -1
        yaku_list.append("清老头")
    # 10. 小四喜/大四喜
    wind_tiles = {"1z", "2z", "3z", "4z"}
    triplet_count = 0
    pair_found = False
    for tile in wind_tiles:
        count = get_tile_counts(alls_tiles).get(tile, 0)
        if count >= 3:
            triplet_count += 1
        elif count == 2:
            pair_found = True
    if triplet_count == 3 and pair_found:
        fan += -1
        yaku_list.append("小四喜")
    if triplet_count == 4:
        fan += -2
        yaku_list.append("大四喜")
    # 11. 四杠子
    kan_count = sum(1 for m in outer_melds if m["type"] in ["ankan","minkan"])
    if kan_count == 4:
        fan += -1
        yaku_list.append("四杠子")
    # 有役满
    if fan < 0:
        return fan,yaku_list
    """standard"""
    # 12. 立直/两立直
    if context.get("isReach", False):
        if context.get("isWReach",False):
            fan += 2
            yaku_list.append("两立直 2")
        else:
            fan += 1
            yaku_list.append("立直 1")
    # 13. 一发
    if context.get("isYiFa", False):fan += 1;yaku_list.append("一发 1")
    # 14. 门前清自摸和
    if context.get("isTsumo", False) and is_menzen:fan += 1;yaku_list.append("门前清自摸和 1")
    # 15. 平和
    if is_pinfu(): fan += 1; yaku_list.append("平和 1")
    # 16. 岭上
    if context.get("isLingShang", False):fan += 1;yaku_list.append("岭上开花 1")
    # 17. 枪杠
    if context.get("isQiangGang", False): fan += 1;yaku_list.append("枪杠 1")
    # 18. 海底
    if context.get("haidi", False): fan += 1;yaku_list.append("海底捞月 1")
    # 19. 河底
    if context.get("hedi", False): fan += 1;yaku_list.append("河底摸鱼 1")
    # 20. 场风牌
    if alls_tiles_counts.get(f"{context['placewind']+1}z", 0) == 3: fan += 1;yaku_list.append("场风牌 1")
    # 21. 门风牌
    if alls_tiles_counts.get(f"{context['selfwind']+1}z", 0) == 3: fan += 1;yaku_list.append("门风牌 1")
    # 22. 一杯口
    def is_iipeikou():
        suits = {'m','p','s'}
        for suit in suits:
            for start in range(1,8):
                seq_tiles = [f"{start}{suit}", f"{start+1}{suit}", f"{start+2}{suit}"]
                if all(all_tiles_counts.get(t,0) >= 2 for t in seq_tiles):
                    temp_counts = all_tiles_counts.copy()
                    for t in seq_tiles:
                        temp_counts[t] -= 2
                        if temp_counts[t] == 0:
                            del temp_counts[t]
                    if check_standard_hand(temp_counts,PATTERNS_3N,PATTERNS_3NP2):
                        return True
        return False
    
    if len(outer_melds)==0 and decomposition.get("type") == "standard":
        if check_standard_hand(all_tiles_counts,PATTERNS_liangbeikou,PATTERNS_liangbeikoup2):
            fan += 3; yaku_list.append("两杯口 3")
        elif is_iipeikou():fan += 1; yaku_list.append("一杯口 1")
    # 23. 断幺九
    if is_tanyao(): fan += 1; yaku_list.append("断幺九 1")
    # 23. 三元牌白/发/中
    if alls_tiles_counts.get(f"5z", 0) == 3: fan += 1;yaku_list.append("白 1")
    if alls_tiles_counts.get(f"6z", 0) == 3: fan += 1;yaku_list.append("发 1")
    if alls_tiles_counts.get(f"7z", 0) == 3: fan += 1;yaku_list.append("中 1")
    # 24. 三色同刻(没有三色同顺)
    def is_sanshoku_doukou(tile_counts: collections.Counter) -> bool:
        """
        三色同刻判定（三麻规则，仅检查1/9）
        :param tile_counts: 牌计数器(Counter对象)
        :return: 是否满足三色同刻条件
        """
        # 定义需要检查的数字和花色
        target_numbers = {'1', '9'}
        required_suits = {'m', 'p', 's'}  # 三麻一般保留万/筒/索
        # 按数字和花色建立统计结构
        num_suit_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        # 过滤并统计有效牌
        for tile, count in tile_counts.items():
            num = tile[0]
            suit = tile[1]
            if num in target_numbers and suit in required_suits:
                num_suit_counts[num][suit] = count
        # 检查每个目标数字
        for num in target_numbers:
            # 需要三个不同花色都至少有3张
            valid_suits = [suit for suit, cnt in num_suit_counts[num].items()if cnt >= 3]
            if len(valid_suits) >= 3:
                return True
        return False
    if is_sanshoku_doukou(alls_tiles_counts): fan += 2;yaku_list.append("三色同刻 2")
    # 25. 三杠子
    if kan_count == 3: fan += 2;yaku_list.append("三杠子 2")
    # 26. 对对和
    jantou = check_counts(alls_tiles_counts)
    if jantou: fan += 2;yaku_list.append("对对和 2")
    # 27. 三暗刻
    if "两杯口 3" not in yaku_list:
        closed_ankou = sum(1 for cnt in all_tiles_counts.values() if cnt >= 3)
        open_ankou = sum(1 for meld in outer_melds if meld.get("type") == "ankan")
        if (closed_ankou + open_ankou) >= 3:
            yaku_list.append("三暗刻 2")
            fan += 2
            if "一杯口 1" in yaku_list:yaku_list.remove("一杯口 1");fan-=1
    # 28. 小三元
    sangen = {"5z", "6z", "7z"}
    triplet_count = 0
    pair_found = False
    for tile in sangen:
        count = get_tile_counts(alls_tiles).get(tile, 0)
        if count >= 3:
            triplet_count += 1
        elif count == 2:
            pair_found = True
    if triplet_count == 2 and pair_found:
        fan += 2
        yaku_list.append("小三元 2")
    # 29. 混老头
    yaojiu = {"1m","9m","1s","9s","1p","9p","1z","2z","3z","4z","5z", "6z", "7z"}
    if all(tile in yaojiu for tile in alls_tiles_counts):
        fan += 2
        yaku_list.append("混老头 2")
    # 30. 七对子
    if is_menzen and "两杯口 3" not in yaku_list:
        if win_type == "chiitoitsu":fan += 2, yaku_list.append("Chiitoitsu")
    # 31. 纯全/混全带幺九
    def is_chanta(outer_melds: List[Dict], pure: bool) -> bool:
        """
        混全带幺九判定函数
        :param all_tiles: 未副露的手牌列表（已包含红宝牌转换）
        :param outer_melds: 副露面子列表
        :return: 是否满足混全带幺九条件
        """
        # 辅助函数：判断面子是否有效
        def is_valid_meld(meld_tiles: List[str]) -> bool:
            # 刻子检查
            if len(set(meld_tiles)) == 1:
                tile = meld_tiles[0]
                return (tile[1] == 'z' and (not pure)) or tile[0] in ['1', '9']
            # 顺子检查（必须123或789）
            nums = sorted(int(t[0]) for t in meld_tiles)
            return nums in [[1,2,3], [7,8,9]]
        for meld in outer_melds:
            if not is_valid_meld(meld.get("tiles", [])):
                return False
        for tile in alls_tiles:
            num, suit = tile[0], tile[1]
            if (suit == 'z' and pure) or (num in ['4', '5','6'] and suit != "z"):
                return False
        # 步骤2：生成候选雀头
        temp_counts = all_tiles_counts.copy()
        # 有效雀头候选（幺九或字牌）
        candidates = [tile for tile, cnt in temp_counts.items() 
                    if cnt >= 2 and (tile[0] in ['1','9'] or tile[1] == 'z')]
        # 步骤3：遍历所有雀头可能性
        def check_remaining(counts: collections.Counter) -> bool:
            """检查剩余牌是否能组成4个有效面子"""
            if sum(counts.values()) == 0:
                return True
            for tile in list(counts.keys()):
                # 尝试顺子（123/789）
                if tile[1] != 'z':
                    num = int(tile[0])
                    suit = tile[1]
                    # 检查可能的顺子组合
                    for start in [1,7]:
                        if num == start:
                            seq = [f"{n}{suit}" for n in [start, start+1, start+2]]
                            if all(counts.get(t,0) >= 1 for t in seq):
                                new_counts = counts.copy()
                                for t in seq:
                                    new_counts[t] -= 1
                                    if new_counts[t] == 0:del new_counts[t]
                                if check_remaining(new_counts):
                                    return True
                # 尝试刻子
                if counts[tile] >= 3:
                    new_counts = counts.copy()
                    new_counts[tile] -= 3
                    if new_counts[tile] == 0:
                        del new_counts[tile]
                    if check_remaining(new_counts):
                        return True
            return False
        for jantou in candidates:
            temps_counts = temp_counts.copy()
            temps_counts[jantou] -= 2
            if temps_counts[jantou] == 0: del temps_counts[jantou]
            # 步骤4：检查剩余牌能否组成4个有效面子
            if check_remaining(temps_counts):
                return True
        return False
    if is_chanta(outer_melds, True): 
        fan += 3 - (not is_menzen);yaku_list.append(f"纯全带幺九 {3 - (not is_menzen)}")
    elif is_chanta(outer_melds, False):
        fan += 2 - (not is_menzen);yaku_list.append(f"混全带幺九 {2 - (not is_menzen)}")
    # 32. 一气通贯
    def is_ittsuu():
        total_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for tile, count in all_tiles_counts.items():
            if tile[-1] not in ['m', 'p', 's']:
                continue  # 跳过字牌
            num = int(tile[0])
            suit = tile[1]
            total_counts[suit][num] += count
        temp_counts = all_tiles_counts.copy()
        for suit in ['m', 'p', 's']:
            counts = total_counts[suit]
            has_123 = all(counts.get(num, 0) >= 1 for num in [1, 2, 3])
            has_456 = all(counts.get(num, 0) >= 1 for num in [4, 5, 6])
            has_789 = all(counts.get(num, 0) >= 1 for num in [7, 8, 9])
            if has_123 and has_456 and has_789:
                for t in [f"{i}{suit}" for i in range(1,10)]:
                    temp_counts[t] -= 1
                    if temp_counts[t] == 0:
                        del temp_counts[t]
                if check_standard_hand(temp_counts,PATTERNS_3N,PATTERNS_3NP2):
                    return True
        return False
    if is_ittsuu():fan += 2 - (not is_menzen);yaku_list.append(f"一气通贯 {2 - (not is_menzen)}")
    # 33. 混一色/清一色
    def yise():
        allstiles = alls_tiles.copy()
        has_honor = False
        numeric_suits = set()
        suit_counter = collections.defaultdict(int)
        for tile in allstiles:
            suit = tile[1]
            if suit == 'z':
                has_honor = True
            else:
                numeric_suits.add(suit)
                suit_counter[suit] += 1
        num_numeric_suits = len(numeric_suits)
        return has_honor, num_numeric_suits
    hashonor,  num_numeric_suits = yise()
    if num_numeric_suits == 1:
        if hashonor:fan += 3 - (not is_menzen);yaku_list.append(f"混一色 {3 - (not is_menzen)}")
        else:fan += 6 - (not is_menzen);yaku_list.append(f"清一色 {6 - (not is_menzen)}")
    
    # 3. Dora calculation
    all_hand_tiles = alls_tiles + ["4z"]*context.get("beidora", 0)
    dora_indicators = parse_tiles(context.get("dora", ""))
    inner_dora_indicators = parse_tiles(context.get("innerdora", ""))
    ndora,ninnerdora,nreddora,nbeidora = 0,0,0,0
    for dora in dora_indicators:
        ndora += sum(1 for tile in all_hand_tiles if tile==dora)
    if context.get("isReach", False):
        for dora in inner_dora_indicators:
            ninnerdora += sum(1 for tile in all_hand_tiles if tile == dora)
    def count_red_balls(outer_str):
        # 分割副露字符串
        units = outer_str.split()
        red_ball_count = 0
        for unit in units:
            if unit.startswith('0') and unit[-2]=='0' and len(unit) == 5:
                # 暗杠单元，形式为 0xx0m
                num = unit[2]  # 获取数字部分
                suit = unit[-1]  # 获取花色部分
                if num == '5' and suit != 'z':
                    red_ball_count += 1
            else:
                # 非暗杠单元，直接统计 '0' 的个数
                red_ball_count += unit.count('0')
        return red_ball_count
    pilestr = context.get("inner","")+ context.get("jinzhang","")
    outer_str = context.get("outer","")
    nreddora += (count_red_balls(outer_str) + pilestr.count("0"))
    nbeidora += context.get("beidora", 0)
    if not context.get("isReach", False): ninnerdora =0
    dora_count = ndora+ninnerdora+nreddora+nbeidora
    if fan > 0:
        fan += dora_count
        if ndora: yaku_list.append(f"宝牌 {ndora}")
        if context.get("isReach", False): yaku_list.append(f"里宝牌 {ninnerdora}")
        if nreddora: yaku_list.append(f"红宝牌 {nreddora}")
        if nbeidora: yaku_list.append(f"北宝牌 {nbeidora}")
    return fan, yaku_list


# --- Main Evaluation Function ---

def evaluate_hand(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    手牌评估函数，包含副露解析和牌数验证
    """
    global PATTERNS_3N, PATTERNS_3NP2, PATTERNS_shuntsu,PATTERNS_shuntsup2
    global PATTERNS_liangbeikou,PATTERNS_liangbeikoup2
     # 解析副露牌组
    def parse_outer_melds(outer_str: str) -> Tuple[List[Dict], int]:
        """解析副露字符串为面子列表，返回(面子列表, 杠的数量)"""
        melds = []
        num_kan = 0
        
        for meld_str in outer_str.split():
            # 解析暗杠 0330m → 3m暗杠(包含红宝牌)
            if meld_str.startswith('0') and meld_str[-2]=='0' and len(meld_str) == 5:
                suit = meld_str[-1]
                num = meld_str[2]  # 中间两位是数字
                
                tiles = []
                for _ in range(3):
                    tiles.append(f"{num}{suit}")
                melds.append({"type": "ankan", "tiles": tiles})
                num_kan += 1
            
            # 解析普通杠 4444m → 明杠
            elif len(meld_str) == 5:
                parsed = parse_tiles(meld_str)
                num = parsed[0]
                suit = parsed[-1]
                melds.append({"type": "minkan", "tiles": [f"{num}{suit}"]*3})
                num_kan += 1
            
            # 解析刻子
            else:  
                parsed = parse_tiles(meld_str)
                if len(parsed) == 3:
                    melds.append({"type": "minkou", "tiles": parsed})
        
        return melds, num_kan

    inner_tiles = parse_tiles(input_data.get("inner", ""))
    jinzhang = parse_tiles(input_data.get("jinzhang", ""))
    outer_melds, num_kan = parse_outer_melds(input_data.get("outer", ""))

    #print(inner_tiles, jinzhang, outer_melds)

    context = input_data # Pass the whole dict for context

    # 1. Prepare Hand Representation
    # 计算总牌数 (杠按3张计算)
    num_outer = sum(3 for meld in outer_melds)
    total_tiles = len(inner_tiles) + len(jinzhang) + num_outer
    if total_tiles != 14:
        return {
            "win": False,
            "reason": f"Invalid total tiles: {total_tiles} (must be 14)"
        }
    
    # 合并手牌计数
    all_closed_tiles = inner_tiles + jinzhang
    closed_counts = get_tile_counts(all_closed_tiles)
    # print(inner_tiles,jinzhang,outer_melds)
    # 2. Check for Winning Hand
    win_decomposition = check_standard_hand(closed_counts.copy(),PATTERNS_3N,PATTERNS_3NP2) # Use copy as check might modify counts
    # print(win_decomposition)
    if (not win_decomposition) and (not outer_melds):
        win_decomposition = check_special_hands(closed_counts.copy()) # Use copy
    if win_decomposition:
        fan, yaku_list = calculate_fan(win_decomposition, context, outer_melds, all_closed_tiles)
        pinhe  = "平和" in yaku_list
        fu = calculate_fu(win_decomposition, context, outer_melds, all_closed_tiles, pinhe)
        return {
            "win": fan!=0,
            "decomposition": win_decomposition, # Include how the hand was broken down
            "yaku": yaku_list,
            "fan": fan,
            "fu": fu,
        }
    else:
        #print("DEBUG: No winning decomposition found.")
        return {"win": False, "reason": "Hand does not form a winning shape."}

# --- Example Usage ---
if __name__ == "__main__":
    test_input = {
        "inner": "'7s0p6p1z4p1z8s",  # Example hand (missing East wind for pair)
        "jinzhang": "9s",          # Drawing the East wind completes pair and triplet
        "outer": "777z 999p",               # No open melds
        "selfwind": 2,             # East
        "placewind": 0,            # East
        "dora": "5s8p",
        "innerdora": "2s7p",
        "beidora": 1,
        "isReach": False,
        "isWReach": False,
        "isYiFa": False,
        "isTsumo": True,
        "haidi": False,
        "hedi": False,
        "isLingShang": False,
        "isQiangGang": False,
        "tianhe": False,
        "dihe": False
    }
    load_patterns()
    print("--- Evaluating Hand ---")
    print(json.dumps(test_input, indent=4))
    result = evaluate_hand(test_input)
    print("\n--- Result ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))
