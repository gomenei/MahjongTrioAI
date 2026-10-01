from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import pygame

from mahjong_env.tile import MeldType, Suit, Tile, Wind


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 760
PANEL_WIDTH = 250
TABLE_WIDTH = SCREEN_WIDTH - PANEL_WIDTH

HAND_TILE = (50, 70)
SMALL_TILE = (35, 49)
MELD_TILE = (29, 41)
TILE_GAP = 3

TABLE_GREEN = (22, 104, 77)
TABLE_DARK = (13, 72, 57)
PANEL_BG = (24, 29, 34)
WHITE = (240, 244, 242)
MUTED = (174, 190, 184)
GOLD = (244, 198, 72)
CYAN = (73, 209, 196)
RED = (229, 91, 91)


def tile_asset_name(tile: Tile) -> str:
    suit_suffix = {
        Suit.Manzu: "m",
        Suit.Pinzu: "p",
        Suit.Souzu: "s",
        Suit.Honors: "z",
    }
    if tile.is_red and tile.value == 5 and tile.suit in {Suit.Manzu, Suit.Pinzu, Suit.Souzu}:
        return f"0{suit_suffix[tile.suit]}"
    return f"{tile.value}{suit_suffix[tile.suit]}"


class MahjongGUI:
    def __init__(self, asset_dir: Path):
        pygame.init()
        pygame.display.set_caption("三人日麻 AI 对局")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.asset_dir = Path(asset_dir)

        self.font_small = self._load_font(17)
        self.font = self._load_font(21)
        self.font_large = self._load_font(31)
        self.font_title = self._load_font(38)

        self.base_images: Dict[str, pygame.Surface] = {}
        self.image_cache: Dict[Tuple, pygame.Surface] = {}
        self.hand_rects: List[Tuple[pygame.Rect, Tile, bool]] = []
        self.action_buttons: List[Tuple[pygame.Rect, str]] = []
        self.nav_buttons: List[Tuple[pygame.Rect, str]] = []
        self.timeline_rect = pygame.Rect(0, 0, 0, 0)
        self.dragging_timeline = False
        self.restart_rect = pygame.Rect(0, 0, 0, 0)
        self.hovered_hand = -1
        self.selection_kind: Optional[str] = None
        self.notice = ""
        self.last_state_signature = None
        self.motion_animations: List[Dict] = []
        self.banner_started = 0
        self.last_banner_text = ""
        self._load_all_tiles()

    @staticmethod
    def _load_font(size: int):
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if font_path.exists():
            return pygame.font.Font(str(font_path), size)
        return pygame.font.SysFont(["Microsoft YaHei", "SimHei", "Arial"], size)

    def _load_all_tiles(self):
        for path in self.asset_dir.glob("*.png"):
            if len(path.stem) == 2 and path.stem[0].isdigit():
                self.base_images[path.stem] = pygame.image.load(str(path)).convert_alpha()

    def _tile_image(
        self, tile: Tile, size: Tuple[int, int], rotation: int = 0
    ) -> pygame.Surface:
        name = tile_asset_name(tile)
        key = (name, size, rotation)
        if key not in self.image_cache:
            image = pygame.transform.smoothscale(self.base_images[name], size)
            if rotation:
                image = pygame.transform.rotate(image, rotation)
            self.image_cache[key] = image
        return self.image_cache[key]

    def _draw_tile(
        self,
        tile: Tile,
        x: int,
        y: int,
        size: Tuple[int, int] = SMALL_TILE,
        rotation: int = 0,
        border: Optional[Tuple[int, int, int]] = None,
    ) -> pygame.Rect:
        image = self._tile_image(tile, size, rotation)
        rect = image.get_rect(topleft=(x, y))
        self.screen.blit(image, rect)
        if tile.is_red:
            pygame.draw.rect(self.screen, RED, rect, 2, border_radius=3)
        if border:
            pygame.draw.rect(self.screen, border, rect.inflate(4, 4), 2, border_radius=4)
        return rect

    def _draw_back(self, x: int, y: int, size, rotation=0):
        width, height = size
        if abs(rotation) == 90:
            width, height = height, width
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (231, 224, 194), rect, border_radius=5)
        inner = rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, (34, 91, 137), inner, border_radius=3)
        pygame.draw.rect(self.screen, (103, 160, 193), inner, 2, border_radius=3)
        return rect

    def handle_events(self, controller):
        mouse_pos = pygame.mouse.get_pos()
        self.hovered_hand = -1
        for index, (rect, _, selectable) in enumerate(self.hand_rects):
            if selectable and rect.collidepoint(mouse_pos):
                self.hovered_hand = index

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if controller.mode == "replay":
                    if event.key == pygame.K_LEFT:
                        controller.replay_change_step(-1)
                    elif event.key == pygame.K_RIGHT:
                        controller.replay_change_step(1)
                    elif event.key == pygame.K_UP:
                        controller.replay_change_round(-1)
                    elif event.key == pygame.K_DOWN:
                        controller.replay_change_round(1)
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        controller.replay_set_view(event.key - pygame.K_1)
                if event.key == pygame.K_ESCAPE:
                    if self.selection_kind:
                        self.selection_kind = None
                    else:
                        self.running = False
                elif event.key == pygame.K_r and controller.done:
                    controller.reset()
                    self.selection_kind = None
                    self.notice = ""
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if controller.mode == "replay" and self.timeline_rect.collidepoint(event.pos):
                    self.dragging_timeline = True
                    self._seek_timeline(event.pos[0], controller)
                elif not self._handle_nav_click(event.pos, controller):
                    self._handle_click(event.pos, controller)
            elif event.type == pygame.MOUSEMOTION and self.dragging_timeline:
                self._seek_timeline(event.pos[0], controller)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging_timeline = False

    def _handle_nav_click(self, pos, controller):
        for rect, command in self.nav_buttons:
            if not rect.collidepoint(pos):
                continue
            if command == "live":
                controller.set_live_mode()
            elif command == "replay":
                if controller.replay:
                    controller.set_replay_mode()
                else:
                    self._open_replay(controller)
            elif command == "load":
                self._open_replay(controller)
            elif command == "prev_step":
                controller.replay_change_step(-1)
            elif command == "next_step":
                controller.replay_change_step(1)
            elif command == "prev_round":
                controller.replay_change_round(-1)
            elif command == "next_round":
                controller.replay_change_round(1)
            elif command.startswith("view_"):
                controller.replay_set_view(int(command[-1]))
            return True
        return False

    def _open_replay(self, controller):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="选择雀魂牌谱 JSON",
                initialdir=str(controller.replay_dir),
                filetypes=[("雀魂牌谱", "*.json"), ("所有文件", "*.*")],
            )
            root.destroy()
            if path and not controller.load_replay(Path(path)):
                self.notice = f"读取失败：{controller.replay_error}"
            elif path:
                self.notice = ""
                self.last_state_signature = None
        except Exception as exc:
            self.notice = f"无法打开文件选择器：{exc}"

    def _seek_timeline(self, mouse_x, controller):
        if not controller.replay:
            return
        ratio = (mouse_x - self.timeline_rect.left) / max(1, self.timeline_rect.width)
        total = len(controller.replay.rounds[controller.replay.round_index])
        controller.replay_seek(round(max(0.0, min(1.0, ratio)) * (total - 1)))

    def _handle_click(self, pos, controller):
        if controller.done:
            if self.restart_rect.collidepoint(pos):
                controller.reset()
                self.selection_kind = None
                self.notice = ""
            return

        if not controller.human_can_act:
            return

        for rect, kind in self.action_buttons:
            if not rect.collidepoint(pos):
                continue
            if kind == "CancelSelection":
                self.selection_kind = None
                self.notice = "已取消立直选择"
                return
            actions = controller.actions_for_kind(kind)
            if kind in {"Riichi", "AnKang", "BuKang"}:
                if self.selection_kind == kind:
                    self.selection_kind = None
                    self.notice = "已取消当前选择"
                else:
                    self.selection_kind = kind
                    self.notice = "请点击要操作的牌；再次点击按钮或按 Esc 可以取消"
            elif actions:
                controller.submit_human_action(actions[0])
                self.selection_kind = None
                self.notice = ""
            return

        for rect, tile, selectable in self.hand_rects:
            if not rect.collidepoint(pos) or not selectable:
                continue
            kind = self.selection_kind or "Play"
            action = controller.tile_action(kind, tile)
            if action is not None and controller.submit_human_action(action):
                self.selection_kind = None
                self.notice = ""
            else:
                self.notice = "这张牌当前不能执行该操作"
            return

    def render(self, controller):
        state = controller.public_state()
        if not controller.human_can_act:
            self.selection_kind = None

        self._sync_animations(state, controller)
        self._draw_background()
        self._draw_center_info(state)
        self._draw_discards(state)
        self._draw_melds(state)
        self._draw_opponent_hands(state)
        self._draw_player_hand(state, controller)
        self._draw_side_panel(state, controller)
        self._draw_motion_animations()
        self._draw_action_banner(controller.last_action if controller.mode == "live" else state.get("event", ""))
        if state["done"] and controller.mode == "live":
            self._draw_result(state)

        pygame.display.flip()
        self.clock.tick(60)

    def _sync_animations(self, state, controller):
        signature = (
            controller.mode,
            state.get("replay_round", 0),
            state.get("event_id", 0),
            tuple(len(river) for river in state["discards"]),
            tuple(len(hand) for hand in state["hands"]),
        )
        if self.last_state_signature is not None and signature != self.last_state_signature:
            previous_lengths = self.last_state_signature[3]
            for seat in range(3):
                if len(state["discards"][seat]) > previous_lengths[seat]:
                    tile = state["discards"][seat][-1]
                    starts = [(TABLE_WIDTH // 2, 650), (TABLE_WIDTH // 2, 80), (80, 360)]
                    ends = [(515, 464), (595, 222), (330, 360)]
                    relative_seat = (seat - state.get("view_player", 0)) % 3
                    self.motion_animations.append({
                        "tile": tile,
                        "start": starts[relative_seat],
                        "end": ends[relative_seat],
                        "rotation": [0, 180, -90][relative_seat],
                        "started": pygame.time.get_ticks(),
                        "duration": 340,
                    })
        self.last_state_signature = signature

        banner = controller.last_action if controller.mode == "live" else state.get("event", "")
        if banner != self.last_banner_text:
            self.last_banner_text = banner
            self.banner_started = pygame.time.get_ticks()

    def _draw_motion_animations(self):
        now = pygame.time.get_ticks()
        active = []
        for animation in self.motion_animations:
            progress = (now - animation["started"]) / animation["duration"]
            if progress >= 1:
                continue
            progress = 1 - (1 - max(0, progress)) ** 3
            x = int(animation["start"][0] + (animation["end"][0] - animation["start"][0]) * progress)
            y = int(animation["start"][1] + (animation["end"][1] - animation["start"][1]) * progress)
            self._draw_tile(animation["tile"], x, y, SMALL_TILE, animation["rotation"], GOLD)
            active.append(animation)
        self.motion_animations = active

    def _draw_action_banner(self, text):
        age = pygame.time.get_ticks() - self.banner_started
        if not text or age > 1150:
            return
        alpha = 255 if age < 750 else int(255 * (1150 - age) / 400)
        rendered = self.font.render(str(text), True, WHITE)
        padding = 18
        surface = pygame.Surface((rendered.get_width() + padding * 2, 44), pygame.SRCALPHA)
        surface.fill((12, 29, 27, min(210, alpha)))
        rendered.set_alpha(alpha)
        surface.blit(rendered, (padding, (44 - rendered.get_height()) // 2))
        rect = surface.get_rect(center=(TABLE_WIDTH // 2, 150))
        self.screen.blit(surface, rect)

    def _draw_background(self):
        self.screen.fill(PANEL_BG)
        table_rect = pygame.Rect(0, 0, TABLE_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, TABLE_GREEN, table_rect)
        pygame.draw.rect(self.screen, TABLE_DARK, table_rect, 8)

        center = (TABLE_WIDTH // 2, SCREEN_HEIGHT // 2 - 8)
        pygame.draw.polygon(
            self.screen,
            (19, 88, 68),
            [(center[0], 122), (846, 570), (188, 570)],
        )
        pygame.draw.polygon(
            self.screen,
            (56, 137, 106),
            [(center[0], 122), (846, 570), (188, 570)],
            2,
        )

    def _draw_center_info(self, state):
        rect = pygame.Rect(TABLE_WIDTH // 2 - 133, 280, 266, 158)
        pygame.draw.rect(self.screen, (20, 42, 43), rect, border_radius=10)
        pulse = (math.sin(pygame.time.get_ticks() / 260) + 1) / 2
        border_color = (
            int(GOLD[0] * (0.72 + 0.28 * pulse)),
            int(GOLD[1] * (0.72 + 0.28 * pulse)),
            int(GOLD[2] * (0.72 + 0.28 * pulse)),
        )
        pygame.draw.rect(self.screen, border_color, rect, 2 + int(pulse), border_radius=10)

        wind_names = {Wind.East: "东", Wind.South: "南", Wind.West: "西", Wind.North: "北"}
        view = state.get("view_player", 0)
        seats = [view, (view + 1) % 3, (view + 2) % 3]
        names = state.get("player_names", ["你", "AI 1", "AI 2"])
        scores = state.get("scores", [35000, 35000, 35000])
        seat_winds = state.get("seat_winds", [Wind.East, Wind.South, Wind.West])

        center_panel = pygame.Rect(rect.centerx - 41, rect.centery - 34, 82, 68)
        pygame.draw.rect(self.screen, (10, 63, 60), center_panel, border_radius=7)
        round_number = state["round_number"]
        if state.get("round_number_zero_based", False):
            round_number += 1
        title = f"{wind_names[state['prevailing_wind']]}{round_number}局"
        self._text(title, self.font_small, GOLD, center_panel.centerx, center_panel.y + 16, center=True)
        self._text(f"余 {state['remaining']}", self.font_small, WHITE, center_panel.centerx, center_panel.y + 45, center=True)

        positions = [
            (rect.centerx, rect.bottom - 15),
            (rect.centerx, rect.y + 15),
            (rect.x + 46, rect.centery),
        ]
        for relative, seat in enumerate(seats):
            name = str(names[seat])
            limit = 4 if relative == 2 else 6
            if len(name) > limit:
                name = name[:limit - 1] + "…"
            wind = wind_names[seat_winds[seat]]
            color = CYAN if seat == state["current_player"] else WHITE
            riichi = " 立" if state["riichi"][seat] else ""
            if relative == 2:
                self._text(f"{wind} {name}{riichi}", self.font_small, color, positions[relative][0], positions[relative][1] - 11, center=True)
                self._text(f"{scores[seat]:,}", self.font_small, GOLD, positions[relative][0], positions[relative][1] + 13, center=True)
            else:
                self._text(
                    f"{wind} {name}  {scores[seat]:,}{riichi}",
                    self.font_small,
                    color,
                    positions[relative][0],
                    positions[relative][1],
                    center=True,
                )

        self._text(
            f"{state['honba']} 本场",
            self.font_small,
            MUTED,
            rect.x + 220,
            rect.centery - 11,
            center=True,
        )
        self._text(
            f"供托 {state['riichi_sticks']}",
            self.font_small,
            MUTED,
            rect.x + 220,
            rect.centery + 13,
            center=True,
        )

    def _draw_player_hand(self, state, controller):
        view = state.get("view_player", 0)
        hand = state["hands"][view].copy()
        drawn = state.get("drawn_tiles", [None, None, None])[view]
        if drawn is not None:
            for index in range(len(hand) - 1, -1, -1):
                tile = hand[index]
                if tile.suit == drawn.suit and tile.value == drawn.value and tile.is_red == drawn.is_red:
                    hand.pop(index)
                    break
        tiles = sorted(hand)
        display_tiles = tiles + ([drawn] if drawn is not None else [])
        extra_gap = 14 if drawn is not None else 0
        width = len(display_tiles) * (HAND_TILE[0] + TILE_GAP) - TILE_GAP + extra_gap
        start_x = max(20, (TABLE_WIDTH - width) // 2)
        base_y = SCREEN_HEIGHT - HAND_TILE[1] - 13
        self.hand_rects = []

        selected_kind = self.selection_kind or "Play"
        for index, tile in enumerate(display_tiles):
            is_drawn = drawn is not None and index == len(display_tiles) - 1
            selectable = (
                controller.mode == "live"
                and view == 0
                and controller.human_can_act
                and controller.tile_action(selected_kind, tile) is not None
            )
            raised = selectable and self.hovered_hand == index
            y = base_y - 10 if raised else base_y
            border = CYAN if selectable else GOLD if is_drawn else None
            x = start_x + index * 53 + (extra_gap if is_drawn else 0)
            if is_drawn and pygame.time.get_ticks() - self.banner_started < 280:
                progress = min(1.0, (pygame.time.get_ticks() - self.banner_started) / 280)
                x += int(22 * (1 - progress))
            rect = self._draw_tile(tile, x, y, HAND_TILE, border=border)
            self.hand_rects.append((rect, tile, selectable))

        if controller.mode == "replay":
            name = state.get("player_names", ["玩家 0", "玩家 1", "玩家 2"])[view]
            self._text(f"当前视角：{name}", self.font_small, WHITE, 24, base_y - 33)
        elif controller.human_can_act:
            prompt = {
                "Riichi": "立直：请选择要打出的牌",
                "AnKang": "暗杠：请选择要杠的牌",
                "BuKang": "加杠：请选择要加杠的牌",
            }.get(self.selection_kind, "轮到你：点击发光的手牌即可打出")
            self._text(prompt, self.font_small, WHITE, 24, base_y - 33)

    def _draw_opponent_hands(self, state):
        view = state.get("view_player", 0)
        top_seat = (view + 1) % 3
        left_seat = (view + 2) % 3
        top_count = len(state["hands"][top_seat])
        top_width = top_count * 30 + 5
        start_x = (TABLE_WIDTH - top_width) // 2
        for i in range(top_count):
            self._draw_back(start_x + i * 30, 23, SMALL_TILE, rotation=180)

        left_count = len(state["hands"][left_seat])
        start_y = 185
        for i in range(left_count):
            self._draw_back(24, start_y + i * 26, SMALL_TILE, rotation=90)


    def _draw_discards(self, state):
        view = state.get("view_player", 0)
        bottom_seat = view
        top_seat = (view + 1) % 3
        left_seat = (view + 2) % 3
        riichi_indices = state.get("riichi_discard_indices", [None, None, None])
        for i, tile in enumerate(state["discards"][bottom_seat]):
            row, col = divmod(i, 6)
            is_riichi = i == riichi_indices[bottom_seat]
            self._draw_tile(
                tile,
                405 + col * 38 - (7 if is_riichi else 0),
                448 + row * 52 + (7 if is_riichi else 0),
                rotation=90 if is_riichi else 0,
            )

        for i, tile in enumerate(state["discards"][top_seat]):
            row, col = divmod(i, 6)
            is_riichi = i == riichi_indices[top_seat]
            self._draw_tile(
                tile,
                595 - col * 38 - (7 if is_riichi else 0),
                222 - row * 52 + (7 if is_riichi else 0),
                rotation=90 if is_riichi else 180,
            )

        for i, tile in enumerate(state["discards"][left_seat]):
            row, col = divmod(i, 6)
            is_riichi = i == riichi_indices[left_seat]
            self._draw_tile(
                tile,
                330 - row * 52 + (7 if is_riichi else 0),
                276 + col * 38 - (7 if is_riichi else 0),
                rotation=0 if is_riichi else -90,
            )

    def _draw_melds(self, state):
        view = state.get("view_player", 0)
        bottom, top, left = view, (view + 1) % 3, (view + 2) % 3
        self._draw_horizontal_meld_area(state["packs"][bottom], bottom, 992, 580, 0, from_right=True)
        self._draw_horizontal_meld_area(state["packs"][top], top, 75, 105, 180, from_right=False)
        self._draw_vertical_meld_area(state["packs"][left], left, 91, 565)

        self._draw_pei_area(state["packs"][bottom], 992, 526, 0, horizontal=True, from_right=True)
        self._draw_pei_area(state["packs"][top], 82, 161, 180, horizontal=True)
        self._draw_pei_area(state["packs"][left], 157, 505, -90, horizontal=False)

    def _meld_entries(self, meld, player, base_rotation):
        tiles = list(meld.tiles)
        called_index = None
        if meld.from_player is not None and meld.type in {MeldType.Chi, MeldType.Pon, MeldType.OpenKan}:
            relative = (meld.from_player - player) % 3
            called_index = 0 if relative == 2 else len(tiles) - 1

        entries = []
        for index, tile in enumerate(tiles):
            face_down = meld.type == MeldType.ClosedKan and index in {0, len(tiles) - 1}
            if index == called_index:
                rotation = 90 if base_rotation in {0, 180} else 0
            else:
                rotation = base_rotation
            entries.append((tile, rotation, face_down))
        return entries

    def _draw_horizontal_meld_area(self, melds, player, anchor_x, y, rotation, from_right):
        normal_melds = [meld for meld in melds if meld.type != MeldType.Pei]
        cursor = anchor_x
        sequence = reversed(normal_melds) if from_right else normal_melds
        for meld in sequence:
            entries = self._meld_entries(meld, player, rotation)
            widths = [MELD_TILE[1] if abs(item[1]) == 90 else MELD_TILE[0] for item in entries]
            group_width = sum(widths) + max(0, len(widths) - 1) * 2
            x = cursor - group_width if from_right else cursor
            for tile, tile_rotation, face_down in entries:
                if face_down:
                    rect = self._draw_back(x, y, MELD_TILE, tile_rotation)
                else:
                    rect = self._draw_tile(tile, x, y, MELD_TILE, tile_rotation)
                x += rect.width + 2
            cursor = (cursor - group_width - 10) if from_right else (cursor + group_width + 10)

    def _draw_vertical_meld_area(self, melds, player, x, anchor_y):
        normal_melds = [meld for meld in melds if meld.type != MeldType.Pei]
        cursor = anchor_y
        for meld in reversed(normal_melds):
            entries = self._meld_entries(meld, player, -90)
            heights = [MELD_TILE[1] if item[1] == 0 else MELD_TILE[0] for item in entries]
            group_height = sum(heights) + max(0, len(heights) - 1) * 2
            y = cursor - group_height
            for tile, tile_rotation, face_down in entries:
                if face_down:
                    rect = self._draw_back(x, y, MELD_TILE, tile_rotation)
                else:
                    rect = self._draw_tile(tile, x, y, MELD_TILE, tile_rotation)
                y += rect.height + 2
            cursor -= group_height + 10

    def _draw_pei_area(self, melds, x, y, rotation, horizontal, from_right=False):
        pei_tiles = [tile for meld in melds if meld.type == MeldType.Pei for tile in meld.tiles]
        if not pei_tiles:
            return
        self._text("拔北", self.font_small, GOLD, x, y - 22)
        for index, tile in enumerate(pei_tiles):
            if horizontal:
                tile_x = x - (index + 1) * (MELD_TILE[0] + 3) if from_right else x + index * (MELD_TILE[0] + 3)
                self._draw_tile(tile, tile_x, y, MELD_TILE, rotation)
            else:
                self._draw_tile(tile, x, y - index * (MELD_TILE[0] + 3), MELD_TILE, rotation)

    def _draw_side_panel(self, state, controller):
        x = TABLE_WIDTH
        pygame.draw.rect(self.screen, PANEL_BG, (x, 0, PANEL_WIDTH, SCREEN_HEIGHT))
        self._text("三人日麻", self.font_large, GOLD, x + 20, 17)
        self.nav_buttons = []
        self.action_buttons = []
        live_rect = pygame.Rect(x + 20, 62, 98, 36)
        replay_rect = pygame.Rect(x + 132, 62, 98, 36)
        self._button(live_rect, "AI 对局", controller.mode == "live")
        self._button(replay_rect, "牌谱回放", controller.mode == "replay")
        self.nav_buttons.extend([(live_rect, "live"), (replay_rect, "replay")])

        if controller.mode == "replay" and controller.replay:
            self._draw_replay_panel(state, controller, x)
        else:
            self._draw_live_panel(state, controller, x)

        if self.notice:
            self._wrapped_text(
                self.notice,
                pygame.Rect(x + 20, 646, PANEL_WIDTH - 40, 62),
                self.font_small,
                GOLD,
            )
        footer = "←/→ 上下巡　1/2/3 视角" if controller.mode == "replay" else "Esc 退出　R 重新开始"
        self._text(footer, self.font_small, MUTED, x + 20, 722)

    def _draw_live_panel(self, state, controller, x):
        self._text("宝牌", self.font_small, MUTED, x + 20, 116)
        for i, tile in enumerate(state["dora"]):
            self._draw_tile(tile, x + 20 + i * 42, 143, SMALL_TILE)

        self._text("最近动作", self.font_small, MUTED, x + 20, 213)
        self._wrapped_text(
            controller.last_action,
            pygame.Rect(x + 20, 242, PANEL_WIDTH - 40, 82),
            self.font_small,
            WHITE,
        )
        if controller.model is None:
            self._text("模型未加载：AI 使用随机策略", self.font_small, RED, x + 20, 327)

        if controller.human_can_act and not state["done"]:
            self._text("可选操作", self.font_small, MUTED, x + 20, 365)
            kinds = ["Hu", "Riichi", "Peng", "Kang", "AnKang", "BuKang", "Pei", "Pass"]
            labels = {
                "Hu": "和", "Riichi": "立直", "Peng": "碰", "Kang": "明杠",
                "AnKang": "暗杠", "BuKang": "加杠", "Pei": "拔北", "Pass": "跳过",
            }
            available = [kind for kind in kinds if controller.actions_for_kind(kind)]
            if self.selection_kind == "Riichi":
                available = ["CancelSelection" if kind == "Riichi" else kind for kind in available]
                labels["CancelSelection"] = "取消立直"
            for index, kind in enumerate(available):
                col, row = index % 2, index // 2
                rect = pygame.Rect(x + 20 + col * 108, 397 + row * 54, 98, 44)
                active = self.selection_kind == kind or kind == "CancelSelection"
                self._action_button(rect, labels[kind], kind, active)
                self.action_buttons.append((rect, kind))

    def _draw_replay_panel(self, state, controller, x):
        replay = controller.replay
        load_rect = pygame.Rect(x + 20, 111, 210, 34)
        self._button(load_rect, "读取其他牌谱", False)
        self.nav_buttons.append((load_rect, "load"))
        filename = state["source_file"]
        if len(filename) > 25:
            filename = filename[:22] + "…"
        self._text(filename, self.font_small, MUTED, x + 20, 151)

        self._text("宝牌", self.font_small, MUTED, x + 20, 181)
        for i, tile in enumerate(state["dora"]):
            self._draw_tile(tile, x + 20 + i * 42, 205, SMALL_TILE)

        prev_round = pygame.Rect(x + 20, 268, 42, 34)
        next_round = pygame.Rect(x + 188, 268, 42, 34)
        self._button(prev_round, "◀", False)
        self._button(next_round, "▶", False)
        self.nav_buttons.extend([(prev_round, "prev_round"), (next_round, "next_round")])
        round_text = f"{replay.round_labels[replay.round_index]}  {replay.round_index + 1}/{len(replay.rounds)}"
        self._text(round_text, self.font_small, WHITE, x + 125, 286, center=True)

        self._text("观察视角", self.font_small, MUTED, x + 20, 319)
        for seat in range(3):
            rect = pygame.Rect(x + 20 + seat * 70, 346, 62, 34)
            self._button(rect, str(seat + 1), replay.view_player == seat)
            self.nav_buttons.append((rect, f"view_{seat}"))

        step = state["replay_step"]
        total = state["replay_total"]
        turns = state.get("turn_counts", [0, 0, 0])
        self._text(f"巡目/步骤  {step + 1}/{total}", self.font_small, MUTED, x + 20, 399)
        self._text(f"三家打牌数：{turns[0]} / {turns[1]} / {turns[2]}", self.font_small, WHITE, x + 20, 426)

        self.timeline_rect = pygame.Rect(x + 20, 458, 210, 12)
        pygame.draw.rect(self.screen, (60, 72, 76), self.timeline_rect, border_radius=6)
        ratio = step / max(1, total - 1)
        filled = self.timeline_rect.copy()
        filled.width = max(6, int(self.timeline_rect.width * ratio))
        pygame.draw.rect(self.screen, CYAN, filled, border_radius=6)
        knob_x = self.timeline_rect.left + int(self.timeline_rect.width * ratio)
        pygame.draw.circle(self.screen, WHITE, (knob_x, self.timeline_rect.centery), 8)

        prev_step = pygame.Rect(x + 20, 487, 98, 38)
        next_step = pygame.Rect(x + 132, 487, 98, 38)
        self._button(prev_step, "上一巡", False)
        self._button(next_step, "下一巡", False)
        self.nav_buttons.extend([(prev_step, "prev_step"), (next_step, "next_step")])

        self._text("当前事件", self.font_small, MUTED, x + 20, 548)
        self._wrapped_text(
            state.get("event", ""),
            pygame.Rect(x + 20, 576, 210, 58),
            self.font_small,
            WHITE,
        )

    def _draw_result(self, state):
        shade = pygame.Surface((TABLE_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 165))
        self.screen.blit(shade, (0, 0))
        card = pygame.Rect(TABLE_WIDTH // 2 - 245, 235, 490, 245)
        pygame.draw.rect(self.screen, (28, 39, 42), card, border_radius=18)
        pygame.draw.rect(self.screen, GOLD, card, 3, border_radius=18)
        self._text("本局结束", self.font_title, GOLD, card.centerx, card.y + 44, center=True)
        message = state["result_message"] or "牌局结束"
        self._text(message, self.font, WHITE, card.centerx, card.y + 112, center=True)
        self.restart_rect = pygame.Rect(card.centerx - 92, card.y + 164, 184, 48)
        self._button(self.restart_rect, "再来一局（R）", False)

    def _action_button(self, rect, label, kind, active=False):
        mouse_over = rect.collidepoint(pygame.mouse.get_pos())
        palette = {
            "Hu": ((91, 32, 34), (255, 205, 91)),
            "Riichi": ((91, 52, 22), (255, 201, 74)),
            "CancelSelection": ((79, 37, 37), (255, 132, 105)),
            "Pass": ((37, 52, 70), (132, 176, 210)),
            "Peng": ((27, 70, 69), (91, 208, 186)),
            "Kang": ((27, 70, 69), (91, 208, 186)),
            "AnKang": ((27, 70, 69), (91, 208, 186)),
            "BuKang": ((27, 70, 69), (91, 208, 186)),
            "Pei": ((59, 49, 77), (191, 158, 231)),
        }
        fill, border = palette.get(kind, ((44, 55, 62), GOLD))
        shadow = rect.move(0, 4)
        pygame.draw.rect(self.screen, (8, 14, 17), shadow, border_radius=9)
        if mouse_over or active:
            fill = tuple(min(255, channel + 24) for channel in fill)
        pygame.draw.rect(self.screen, border, rect, border_radius=9)
        inner = rect.inflate(-5, -5)
        pygame.draw.rect(self.screen, fill, inner, border_radius=7)
        pygame.draw.rect(self.screen, (255, 235, 174), inner, 1, border_radius=7)
        color = (255, 239, 185) if kind in {"Hu", "Riichi", "CancelSelection"} else WHITE
        self._text(label, self.font, color, rect.centerx, rect.centery + 1, center=True)

    def _button(self, rect, label, active=False):
        mouse_over = rect.collidepoint(pygame.mouse.get_pos())
        color = (48, 126, 106) if mouse_over or active else (48, 60, 65)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, CYAN if active else (92, 111, 113), rect, 2, border_radius=8)
        self._text(label, self.font_small, WHITE, rect.centerx, rect.centery + 2, center=True)

    def _text(self, text, font, color, x, y, center=False):
        surface = font.render(str(text), True, color)
        rect = surface.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surface, rect)
        return rect

    def _wrapped_text(self, text, rect, font, color):
        lines = []
        current = ""
        for char in str(text):
            candidate = current + char
            if font.size(candidate)[0] <= rect.width:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
        for index, line in enumerate(lines[:3]):
            self._text(line, font, color, rect.x, rect.y + index * (font.get_height() + 4))

    def is_running(self):
        return self.running

    def quit(self):
        pygame.quit()
