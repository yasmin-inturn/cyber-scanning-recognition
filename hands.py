"""Rastreamento das mãos (MediaPipe) e reconhecimento de gestos."""
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# Índices dos 21 pontos da mão no MediaPipe
WRIST, THUMB_TIP = 0, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20
FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGERS = ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Pinça com histerese: fecha abaixo de PINCH_ON, só solta acima de PINCH_OFF
PINCH_ON, PINCH_OFF = 0.33, 0.50

# Mola dos cantos: STIFFNESS alto = segue mais rápido, DAMPING baixo = balança mais
STIFFNESS, DAMPING = 0.30, 0.62


class HandTracker:
    def __init__(self, model_path, max_hands=2):
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ts = 0

    def detect(self, frame_bgr):
        """Retorna uma lista de mãos; cada mão é uma lista de 21 pontos (x, y) em pixels."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # O modo VIDEO exige timestamps sempre crescentes
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        h, w = frame_bgr.shape[:2]
        return [[(int(p.x * w), int(p.y * h)) for p in hand] for hand in result.hand_landmarks]

    def close(self):
        self.landmarker.close()


class Spring:
    """Canto que persegue o dedo como uma mola: suaviza o tremor e dá um leve 'balanço'."""

    def __init__(self):
        self.pos = None
        self.vel = np.zeros(2, np.float32)

    def update(self, target):
        target = np.asarray(target, np.float32)
        if self.pos is None:
            self.pos = target.copy()
        else:
            self.vel = (self.vel + (target - self.pos) * STIFFNESS) * DAMPING
            self.pos = self.pos + self.vel
        return self.pos

    def reset(self):
        self.pos = None
        self.vel = np.zeros(2, np.float32)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def pinch_ratio(pts, finger_tip):
    # Normaliza pelo tamanho da mão, assim funciona perto ou longe da câmera
    hand_size = dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0
    return dist(pts[THUMB_TIP], pts[finger_tip]) / hand_size


def fingers_up(pts):
    """[indicador, médio, anelar, mindinho]: True se o dedo está esticado."""
    return [dist(pts[WRIST], pts[tip]) > dist(pts[WRIST], pts[pip]) * 1.15 for tip, pip in FINGERS]


def is_open_palm(pts):
    return all(fingers_up(pts))


def is_fist(pts):
    return not any(fingers_up(pts))


def is_scroll_pose(pts):
    index, middle, ring, pinky = fingers_up(pts)
    return index and middle and not ring and not pinky


def corner_targets(hands):
    """4 alvos com identidade fixa (cada um é sempre o mesmo dedo)."""
    if len(hands) >= 2:
        left, right = sorted(hands[:2], key=lambda p: p[WRIST][0])
        return [left[INDEX_TIP], right[INDEX_TIP], right[THUMB_TIP], left[THUMB_TIP]]
    (ax, ay), (bx, by) = hands[0][THUMB_TIP], hands[0][INDEX_TIP]
    x1, x2 = sorted((ax, bx))
    y1, y2 = sorted((ay, by))
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def quad_order(pts):
    """Ordem dos cantos em volta do centro, começando pelo de cima/esquerda (TL, TR, BR, BL)."""
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    start = int(np.argmin(pts[order].sum(axis=1)))
    return np.roll(order, -start)


class Pinch:
    """Pinça do polegar com um dedo específico, em qualquer mão.

    Só conta se esse dedo for o mais perto do polegar, pra pinça com o médio
    não disparar também a do anelar.
    """

    def __init__(self, finger_tip):
        self.tip = finger_tip
        self.active = False
        self.hand = None

    def update(self, hands):
        best = None
        for pts in hands:
            ratios = {tip: pinch_ratio(pts, tip) for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)}
            if min(ratios, key=ratios.get) != self.tip:
                continue
            if best is None or ratios[self.tip] < best[0]:
                best = (ratios[self.tip], pts)
        threshold = PINCH_OFF if self.active else PINCH_ON
        self.active = best is not None and best[0] < threshold
        self.hand = best[1] if self.active else None
        return self.active


class Hold:
    """Gesto que precisa ser segurado um tempo pra disparar (evita acionar sem querer)."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.start = None
        self.fired = False

    def update(self, active, now):
        """Retorna (progresso 0..1, disparou_agora)."""
        if not active:
            self.start = None
            self.fired = False
            return 0.0, False
        if self.start is None:
            self.start = now
        progress = min(1.0, (now - self.start) / self.seconds)
        if progress >= 1.0 and not self.fired:
            self.fired = True
            return 1.0, True
        return (0.0 if self.fired else progress), False


class PalmStill:
    """Palma aberta parada (quase sem mexer). Retorna (parada?, centro da palma)."""

    def __init__(self, max_speed=0.012):
        self.max_speed = max_speed
        self.prev = None
        self.speed = 1.0

    def update(self, hands, frame_w):
        palm = next((p for p in hands if is_open_palm(p)), None)
        if palm is None:
            self.prev, self.speed = None, 1.0
            return False, None
        center = np.array(palm[MIDDLE_MCP], np.float32) / frame_w
        if self.prev is not None:
            self.speed = 0.6 * self.speed + 0.4 * float(np.linalg.norm(center - self.prev))
        self.prev = center
        return self.speed < self.max_speed, palm[MIDDLE_MCP]


class SwipeDetector:
    """Palma aberta deslizando rápido pro lado: -1 (esquerda), 1 (direita) ou 0."""

    def __init__(self, distance=0.28, window=0.4, cooldown=1.0):
        self.distance = distance
        self.window = window
        self.cooldown = cooldown
        self.history = deque()
        self.blocked_until = 0.0

    def update(self, hands, frame_w, now):
        palm = next((p for p in hands if is_open_palm(p)), None)
        if palm is None or now < self.blocked_until:
            self.history.clear()
            return 0
        self.history.append((now, palm[MIDDLE_MCP][0] / frame_w))
        while now - self.history[0][0] > self.window:
            self.history.popleft()
        dx = self.history[-1][1] - self.history[0][1]
        if abs(dx) > self.distance:
            self.history.clear()
            self.blocked_until = now + self.cooldown
            return 1 if dx > 0 else -1
        return 0
