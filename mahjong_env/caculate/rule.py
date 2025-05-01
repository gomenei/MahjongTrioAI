import json
import collections
import os
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
PATTERNS_LOADED = False
PATTERN_LOAD_ERROR = False

def load_patterns():
    """Loads the JSON pattern files into global variables."""
    global PATTERNS_3N, PATTERNS_3NP2, PATTERNS_LOADED, PATTERN_LOAD_ERROR
    if PATTERNS_LOADED or PATTERN_LOAD_ERROR:
        return not PATTERN_LOAD_ERROR

    script_dir = os.path.dirname(__file__) # Get directory of the current script
    path_3n = os.path.join(script_dir, '3n_patterns.json')
    path_3np2 = os.path.join(script_dir, '3np2_patterns.json')

    try:
        with open(path_3n, 'r', encoding='utf-8') as f:
            PATTERNS_3N = json.load(f)
        with open(path_3np2, 'r', encoding='utf-8') as f:
            PATTERNS_3NP2 = json.load(f)
        PATTERNS_LOADED = True
        print("DEBUG: Pattern files loaded successfully.")
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


def check_standard_hand(counts: TypingCounter[str]) -> bool:
    """检查给定的牌（用计数表示）是否可以形成标准手牌,3*N+2"""
    if not load_patterns():
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
            elif num == sorted_nums[i-1] + 1: # Continues the current component
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

        patterns_to_use = PATTERNS_3NP2 if needs_comp_pair else PATTERNS_3N
        n = (comp_count - 2) // 3 if needs_comp_pair else comp_count // 3
        n_key = str(n)

        if n < 0: return False # Invalid component size resulted in N<0

        if n_key not in patterns_to_use:
            # The required number of melds (N) doesn't exist in the patterns file
            print(f"DEBUG: N={n_key} not found in {'3np2' if needs_comp_pair else '3n'} patterns for component {comp_seq}")
            return False

        valid_sequences_for_n = patterns_to_use[n_key]
        if comp_seq not in valid_sequences_for_n:
            # The specific sequence for this component isn't listed as valid for N melds
            print(f"DEBUG: Sequence '{comp_seq}' not found in {'3np2' if needs_comp_pair else '3n'} patterns[{n_key}]")
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

def calculate_fu(decomposition: Dict[str, Any], context: Dict[str, Any], outer_melds: List[Dict]) -> int:
    fu = 0
    win_type = decomposition.get("type", "unknown")

    if win_type == "chiitoitsu":
        fu = 25 # Standard Fu for Seven Pairs
        return fu # No further additions typically

    if win_type == "standard":
        fu = 20 # Base Fu

        # --- Placeholder Fu Additions ---
        # Add fu for open/closed state (Menzen Ron: +10, Tsumo: +2 - but Pinfu exception)
        # Add fu for waits (Penchan, Kanchan, Tanki: +2)
        # Add fu for pair (Jantou) value (Yakuhai pair: +2 or +4 for double wind)
        # Add fu for melds (Mentsu)
        #   - Open simple triplet: +2
        #   - Closed simple triplet: +4
        #   - Open terminal/honor triplet: +4
        #   - Closed terminal/honor triplet: +8
        #   - Open simple quad: +8
        #   - Closed simple quad: +16
        #   - Open terminal/honor quad: +16
        #   - Closed terminal/honor quad: +32

        # Example placeholder additions:
        if context.get("isTsumo", False):
            # Pinfu exception needs to be handled here later
             fu += 2 # Base Tsumo Fu (unless Pinfu)
        else: # Ron
             # Menzen Ron (closed hand) exception needs handling
             fu += 10 # Base Ron Fu (add if hand is closed) - NEEDS menzen check

        # Placeholder for meld fu (needs outer_melds info too)
        # fu += ...

        # Round up fu to nearest 10 (except for 25 from Chiitoitsu)
        if fu != 0:
             fu = ((fu + 9) // 10) * 10

    # Kokushi doesn't usually calculate Fu.

    print(f"DEBUG: Placeholder Fu calculation returning: {fu}")
    # Default for framework if no specific rules applied
    return max(fu, 20 if win_type != "chiitoitsu" else 25)


def calculate_fan(decomposition: Dict[str, Any], 
                  context: Dict[str, Any], 
                  outer_melds: List[Dict], 
                  all_tiles: List[str]) -> Tuple[int, List[str]]:
    fan = 0
    yaku_list = []
    is_menzen = len(outer_melds) == 0 # 是否门清
    all

    def has_yakuhai(tile_type: str) -> bool:
        """检查是否包含役牌刻子/雀头"""
        yakuhai = []
        # 三元牌
        if context.get('selfwind', -1) == Wind.EAST.value:
            yakuhai += ["5z", "6z", "7z"]  # 假设5z=白 6z=发 7z=中
        # 场风
        yakuhai.append(f"{context['placewind']+1}z")
        # 自风
        yakuhai.append(f"{context['selfwind']+1}z")
        
        counts = get_tile_counts(all_tiles)
        return any(counts.get(t, 0) >= 3 for t in yakuhai) or decomposition.get("jantou") in yakuhai
    
    def is_tanyao() -> bool:
        """断幺九判断"""
        for tile in all_tiles:
            num, suit = tile[0], tile[1]
            if suit == 'z' or num in ['1', '9']:
                return False
        return True
    
    def is_pinfu() -> bool:
        """平和判断"""
        if not is_menzen: return False
        if decomposition.get("type") != "standard": return False
        # 雀头不能是役牌
        if has_yakuhai(decomposition.get("jantou", "")): return False
        # 所有面子必须是顺子
        return all(m['type'] == 'shuntsu' for m in decomposition.get("mentsu", []))

    win_type = decomposition.get("type", "unknown")

    # --- Placeholder Yaku Checks ---
    # This is where you'd implement checks for all the Yaku (役).

    # 0. Check Yakuman first (Kokushi, Suuankou, Daisangen, etc.)
    if win_type == "kokushi":
        yaku_list.append("Kokushi Musou (十三幺)")
        return 13, yaku_list # Treat Yakuman as 13 fan for calculation ceiling

    # 1. Game state / circumstantial Yaku
    if context.get("isReach", False):
        fan += 1
        yaku_list.append("Reach (立直)")
    if context.get("isWReach", False):
        # Usually replaces Reach fan, check specific rules
        fan += 1 # Adds 1 more fan on top of Reach? Or just 2 total? Assume +1 for now.
        yaku_list.append("Double Reach (两立直)")
    if context.get("isYiFa", False):
        fan += 1
        yaku_list.append("Ippatsu (一发)")
    if context.get("isTsumo", False) and not outer_melds: # Menzen Tsumo check needs closed hand verification
         is_menzen = not outer_melds # Simplistic check, needs refinement
         if is_menzen:
             fan += 1
             yaku_list.append("Menzen Tsumo (门前清自摸和)")
    if context.get("isLingShang", False):
        fan += 1
        yaku_list.append("Rinshan Kaihou (岭上开花)")
    # Add Haidi, Houtei, Chankan, Tenhou, Diho checks here...

    # 2. Yaku based on hand structure (Examples)
    # if is_pinfu(decomposition, context): fan += 1; yaku_list.append("Pinfu (平和)")
    # if is_tanyao(decomposition, outer_melds): fan += 1; yaku_list.append("Tanyao (断幺九)")
    # if is_yakuhai_pair(decomposition.get("jantou"), context): fan += Y; yaku_list.append("Yakuhai (役牌)")
    # if is_sanshoku_doujun(decomposition, outer_melds): fan += Y; yaku_list.append("Sanshoku Doujun (三色同顺)")
    # if is_chanta(decomposition, outer_melds): fan += Y; yaku_list.append("Chantaiyao (混全带幺九)")
    # if is_toitoi(decomposition, outer_melds): fan += 2; yaku_list.append("Toitoi (对对和)")
    # if is_chinitsu(all_tiles): fan += Y; yaku_list.append("Chinitsu (清一色)")
    # ... and many more ...

    if win_type == "chiitoitsu":
        fan += 2
        yaku_list.append("Chiitoitsu (七对子)")

    # 3. Dora calculation
    dora_count = 0
    all_hand_tiles = inner_tiles + [tile for meld in outer_melds for tile in meld.get("tiles", [])]
    dora_indicators = parse_tiles(context.get("dora", ""))
    inner_dora_indicators = parse_tiles(context.get("innerdora", ""))
    # Needs logic to map indicators to actual dora tiles
    # Needs logic to count red dora (from original input strings)
    # Needs logic for Kita dora (beidora)
    # dora_count = count_dora(all_hand_tiles, dora_indicators, inner_dora_indicators, context.get("beidora", 0), red_dora_count)

    if dora_count > 0:
         fan += dora_count
         yaku_list.append(f"Dora x{dora_count} (宝牌{dora_count})")

    # Check for minimum 1 Fan (Yaku Shibari) - except under specific rulesets
    if fan == dora_count and fan > 0 and not any(yaku not in ["Reach (立直)", "Dora"] for yaku in yaku_list):
        # If only dora/reach contributed fan, and no other structural yaku, it might not be a valid win
        # This needs careful implementation based on rules (kuikae, etc.)
        # For now, assume if we got here, there was at least one structural yaku or valid circumstantial one.
        pass

    # Yakuman check again? Or handle potential composite Yakuman scoring?

    print(f"DEBUG: Placeholder Fan calculation returning: {fan}, Yaku: {yaku_list}")
    if fan > 0 and not yaku_list:
         # This indicates only Dora contributed; might be invalid win if no other Yaku.
         # However, circumstantial Yaku like Reach, Tsumo are valid.
         # For the framework, we'll allow it, but a full implementation needs Yaku Shibari check.
         print("WARN: Fan > 0 but no specific Yaku identified (likely Dora only or circumstantial)")
         # We might need to return 0 fan if no actual Yaku exists besides Dora/Reach etc.
         # Let's assume for the framework that if Fan>0, it's valid.
         pass

    # Return fan capped at Kazoe Yakuman (13) if needed by rules, or allow higher.
    # For now, return calculated fan.
    return fan, yaku_list


# --- Main Evaluation Function ---

def evaluate_hand(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    手牌评估函数，包含副露解析和牌数验证
    """
     # 解析副露牌组
    def parse_outer_melds(outer_str: str) -> Tuple[List[Dict], int]:
        """解析副露字符串为面子列表，返回(面子列表, 杠的数量)"""
        melds = []
        num_kan = 0
        
        for meld_str in outer_str.split(" "):
            # 解析暗杠 0330m → 3m暗杠(包含红宝牌)
            if meld_str.startswith('0') and meld_str.endswith('0') and len(meld_str) == 5:
                suit = meld_str[-1]
                num = meld_str[2]  # 中间两位是数字
                
                tiles = []
                for _ in range(3):
                    tiles.append(f"{num}{suit}")
                melds.append({"type": "ankan", "tiles": tiles})
                num_kan += 1
            
            # 解析普通杠 4444m → 明杠
            elif len(meld_str) == 5 and meld_str[0] == meld_str[1] == meld_str[2] == meld_str[3]:
                num = meld_str[0]
                suit = meld_str[-1]
                melds.append({"type": "minkan", "tiles": [f"{num}{suit}"]*3})
                num_kan += 1
            
            # 解析顺子/刻子 234p → 顺子
            else:  
                parsed = parse_tiles(meld_str)
                if len(parsed) == 3:
                    meld_type = "koutsu"
                    melds.append({"type": meld_type, "tiles": parsed})
        
        return melds, num_kan

    inner_tiles = parse_tiles(input_data.get("inner", ""))
    jinzhang = parse_tiles(input_data.get("jinzhang", ""))
    outer_melds, num_kan = parse_outer_melds(input_data.get("outer", ""))
    
    context = input_data # Pass the whole dict for context

    # 1. Prepare Hand Representation
    # 计算总牌数 (杠按3张计算)
    num_outer = sum(
        3 if meld["type"] in ["ankan", "minkan"] else len(meld["tiles"])
        for meld in outer_melds
    )
    total_tiles = len(inner_tiles) + len(jinzhang) + num_outer

    if total_tiles != 14:
        return {
            "win": False,
            "reason": f"Invalid total tiles: {total_tiles} (must be 14)"
        }
    
    # 合并手牌计数
    all_closed_tiles = inner_tiles + jinzhang
    closed_counts = get_tile_counts(all_closed_tiles)
    # 2. Check for Winning Hand
    win_decomposition = None
    # Check special hands first
    if not outer_melds: # Special hands generally require a closed hand
        win_decomposition = check_special_hands(closed_counts.copy()) # Use copy as check might modify counts

    # If not a special hand, check standard hand
    if win_decomposition is None:
        win_decomposition = check_standard_hand(closed_counts.copy()) # Use copy

    # 3. Calculate Score if Winning
    if win_decomposition:
        print(f"DEBUG: Win condition met. Type: {win_decomposition.get('type')}")
        # --- Call Calculation Functions ---
        # Pass the found decomposition, context, parsed outer melds, and tile list
        fan, yaku_list = calculate_fan(win_decomposition, context, outer_melds, all_closed_tiles)

        if fan == 0:
             # Check Yaku Shibari - Must have at least one Yaku (excluding Dora sometimes)
             print("DEBUG: Hand is Agari shape, but calculated Fan is 0 (Yaku Shibari failed).")
             return {"win": False, "reason": "No Yaku found (Yaku Shibari)."}

        fu = calculate_fu(win_decomposition, context, outer_melds)

        # Placeholder for final score calculation based on Fan, Fu, dealer status, tsumo/ron
        final_score = 0 # Needs lookup table: calculate_points(fan, fu, is_dealer, is_tsumo)

        return {
            "win": True,
            "decomposition": win_decomposition, # Include how the hand was broken down
            "yaku": yaku_list,
            "fan": fan,
            "fu": fu,
            "score": final_score, # Placeholder
            "score_details": {} # Placeholder for breakdown (oya/ko, tsumo/ron payments)
        }
    else:
        print("DEBUG: No winning decomposition found.")
        return {"win": False, "reason": "Hand does not form a winning shape."}

# --- Example Usage ---
if __name__ == "__main__":
    test_input = {
        "inner": "111234m123789p1z",  # Example hand (missing East wind for pair)
        "jinzhang": "1z",          # Drawing the East wind completes pair and triplet
        "outer": "",               # No open melds
        "selfwind": 0,             # East
        "placewind": 0,            # East
        "dora": "1p",
        "innerdora": "",
        "beidora": 0,
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

    # This example input doesn't perfectly match the placeholder decomposition,
    # because the placeholder check logic is not real.
    # A real check would find the pair is 1z, and melds are 111m, 234m, 789p, 11s (if 's' was present)
    # Let's adjust input to potentially match the *placeholder* output for demo:
    test_input_matching_placeholder = {
        "inner": "111234m79p11s1z1z1z", # 12 tiles, waiting on 1s for pair?
        "jinzhang": "8p",          # Draw 1s. Pair = 1s, Melds = 111m, 234m, 789p, 111z? Yes.
        "outer": "",
        "selfwind": 0, "placewind": 0, "dora": "1p", "innerdora": "", "beidora": 0,
        "isReach": False, "isWReach": False, "isYiFa": False, "isTsumo": True,
        "haidi": False, "hedi": False, "isLingShang": False, "isQiangGang": False,
        "tianhe": False, "dihe": False
    }


    print("--- Evaluating Hand ---")
    print(json.dumps(test_input, indent=4))
    result = evaluate_hand(test_input)
    print("\n--- Result ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print("--- Evaluating Hand ---")
    print(json.dumps(test_input_matching_placeholder, indent=4))
    result = evaluate_hand(test_input_matching_placeholder)
    print("\n--- Result ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print("\n--- Evaluating Non-Winning Hand (Example) ---")
    non_winning_input = {
        "inner": "123456789m123p1s", # Needs one more tile
        "jinzhang": "4p",          # Doesn't complete hand
        "outer": "",
        "selfwind": 1, "placewind": 0, "dora": "1z", "innerdora": "", "beidora": 0,
        "isReach": False, "isWReach": False, "isYiFa": False, "isTsumo": False,
        "haidi": False, "hedi": False, "isLingShang": False, "isQiangGang": False,
        "tianhe": False, "dihe": False
    }
    print(json.dumps(non_winning_input, indent=4))
    result_non_winning = evaluate_hand(non_winning_input)
    print("\n--- Result ---")
    print(json.dumps(result_non_winning, indent=4, ensure_ascii=False))


