"""Modo MOUSE: controla o cursor do Mac com a mão.

No macOS, o app que roda o script (PyCharm ou Terminal) precisa de permissão em
Ajustes do Sistema > Privacidade e Segurança > Acessibilidade, senão o cursor não se move.
"""
import cv2
import numpy as np

from hands import INDEX_MCP, INDEX_TIP, MIDDLE_TIP, Pinch, is_fist, is_scroll_pose
from hud import AMBER, GREEN, PANEL_X, WHITE, dashed_rect, panel_start, text

# Área da câmera que vira a tela inteira (frações do frame). Menor = menos movimento de braço.
ZONE_X = (0.25, 0.75)
ZONE_Y = (0.18, 0.65)
SCROLL_STEP_PX = 12  # quanto a mão sobe/desce (em pixels da câmera) pra rolar 1 passo


class MouseController:
    def __init__(self):
        import pyautogui  # importa só aqui: precisa das permissões do macOS

        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False  # levar o cursor pro canto não deve derrubar o programa
        self.gui = pyautogui
        self.screen = np.array(pyautogui.size(), np.float32)
        self.pos = None
        self.dragging = False
        self.left = Pinch(INDEX_TIP)
        self.right = Pinch(MIDDLE_TIP)
        self.scroll_anchor = None
        self.status = "NO HAND"

    def release(self):
        if self.dragging:
            self.gui.mouseUp()
            self.dragging = False

    def update(self, hand, w, h):
        if hand is None:
            self.release()
            self.scroll_anchor = None
            self.status = "NO HAND"
            return
        if is_fist(hand):
            self.release()
            self.pos = None
            self.status = "PAUSED"
            return

        if is_scroll_pose(hand):
            self.release()
            y = hand[INDEX_TIP][1]
            if self.scroll_anchor is None:
                self.scroll_anchor = y
            steps = int((self.scroll_anchor - y) / SCROLL_STEP_PX)  # mão sobe -> rola pra cima
            if steps:
                self.gui.scroll(steps)
                self.scroll_anchor = y
            self.status = "SCROLL"
            return
        self.scroll_anchor = None

        # A base do indicador quase não mexe durante a pinça: o clique não tira o cursor do lugar
        x, y = hand[INDEX_MCP]
        nx = np.clip((x / w - ZONE_X[0]) / (ZONE_X[1] - ZONE_X[0]), 0, 1)
        ny = np.clip((y / h - ZONE_Y[0]) / (ZONE_Y[1] - ZONE_Y[0]), 0, 1)
        target = np.array([nx, ny], np.float32) * (self.screen - 1)
        if self.pos is None:
            self.pos = target
        else:
            # Suavização adaptativa: parado = firme, rápido = responde na hora
            alpha = float(np.clip(np.linalg.norm(target - self.pos) / 150, 0.12, 0.7))
            self.pos = self.pos + alpha * (target - self.pos)
        self.gui.moveTo(int(self.pos[0]), int(self.pos[1]))

        was_left = self.left.active
        left = self.left.update([hand])
        if left and not was_left:
            self.gui.mouseDown()
            self.dragging = True
        elif was_left and not left:
            self.release()

        was_right = self.right.active
        if self.right.update([hand]) and not was_right and not left:
            self.gui.rightClick()

        self.status = "CLICK / DRAG" if left else "TRACKING"

    def draw(self, img, hand):
        h, w = img.shape[:2]
        zx0, zx1 = int(ZONE_X[0] * w), int(ZONE_X[1] * w)
        zy0, zy1 = int(ZONE_Y[0] * h), int(ZONE_Y[1] * h)
        dashed_rect(img, (zx0, zy0), (zx1, zy1), GREEN)
        text(img, "CONTROL ZONE", (zx0, zy0 - 8), GREEN, 0.9)
        if hand is not None:
            x, y = hand[INDEX_MCP]
            color = AMBER if self.status == "CLICK / DRAG" else GREEN
            cv2.circle(img, (x, y), 14, color, 1, cv2.LINE_AA)
            cv2.line(img, (x - 22, y), (x + 22, y), color, 1)
            cv2.line(img, (x, y - 22), (x, y + 22), color, 1)

        x0 = PANEL_X
        y = panel_start(img, "REMOTE INPUT // MOUSE CONTROL", 200)
        cursor = "--" if self.pos is None else f"{int(self.pos[0])}, {int(self.pos[1])}"
        rows = [
            (f"STATUS: {self.status}", AMBER if self.status in ("CLICK / DRAG", "PAUSED") else GREEN),
            (f"SCREEN: {int(self.screen[0])} x {int(self.screen[1])}", GREEN),
            (f"CURSOR: {cursor}", WHITE),
            ("", GREEN),
            ("PINCH THUMB+INDEX   click / hold to drag", GREEN),
            ("PINCH THUMB+MIDDLE  right click", GREEN),
            ("INDEX+MIDDLE UP     scroll (move up/down)", GREEN),
            ("FIST                pause // hold 1.5s: exit", GREEN),
        ]
        for i, (s, color) in enumerate(rows):
            text(img, s, (x0, y + 14 + i * 18), color, 0.85)
