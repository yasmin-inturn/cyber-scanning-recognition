"""Modo BIO: leitura facial real com o MediaPipe Face Landmarker (478 pontos + 52 expressões).

Mede o que o rosto FAZ (piscadas, sorriso, direção da cabeça), não o que a pessoa sente.
"""
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from hud import (AMBER, GREEN, PANEL_W, PANEL_X, RED, THUMB_H, THUMB_W, WHITE, bar,
                 corner_brackets, draw_level_meter, halftone, panel_start, text)

NOSE_TIP = 1
SIGNALS = ("SMILE", "BROW RAISE", "JAW OPEN", "EYE OPENNESS", "HEAD MOTION")

# Limites da direção da cabeça (posição do nariz dentro do rosto, -1..1). Ajuste se precisar.
TURN_X = 0.25
LOOK_UP_Y, LOOK_DOWN_Y = -0.05, 0.40

STATE_COLORS = {"FOCUSED": GREEN, "ENGAGED": GREEN, "RESTLESS": AMBER,
                "DISTRACTED": AMBER, "DROWSY": RED, "NO FACE": WHITE}


class FaceTracker:
    def __init__(self, model_path):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_ts = 0

    def detect(self, frame_bgr):
        """Retorna (pontos em pixels, {expressão: 0..1}) ou None se não houver rosto."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in result.face_landmarks[0]]
        blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return pts, blend

    def close(self):
        self.landmarker.close()


class BioReader:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {k: 0.0 for k in SIGNALS}
        self.blinks = deque()
        self.blink_rate = 0.0
        self.eyes_closed = False
        self.closed_since = 0.0
        self.started = None
        self.direction = "--"
        self.away_since = None
        self.prev_nose = None
        self.alert = 0.0
        self.state = "NO FACE"
        self.face_pts = None
        self.bbox = None
        self.last_seen = -10.0
        self.thumb = None
        self.frame_n = 0

    def update(self, face, now, gray):
        self.frame_n += 1
        if face is None:
            self.face_pts = None
            self.prev_nose = None
            if now - self.last_seen > 1.0:
                self.state = "NO FACE"
            for k in self.values:
                self.values[k] *= 0.9
            return

        pts, bs = face
        self.face_pts, self.last_seen = pts, now
        if self.started is None:
            self.started = now

        # Piscadas: olho fecha e abre em menos de meio segundo
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        if not self.eyes_closed and blink > 0.55:
            self.eyes_closed, self.closed_since = True, now
        elif self.eyes_closed and blink < 0.35:
            self.eyes_closed = False
            if now - self.closed_since < 0.5:
                self.blinks.append(now)
        while self.blinks and now - self.blinks[0] > 60:
            self.blinks.popleft()
        window = min(60.0, max(now - self.started, 10.0))
        self.blink_rate = len(self.blinks) * 60 / window

        # Direção da cabeça: onde o nariz está dentro da caixa do rosto
        arr = np.array(pts, np.float32)
        (x0, y0), (x1, y1) = arr.min(axis=0), arr.max(axis=0)
        fw, fh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        self.bbox = (int(x0), int(y0), int(x1), int(y1))
        nose = arr[NOSE_TIP]
        ox = (nose[0] - (x0 + x1) / 2) / (fw / 2)
        oy = (nose[1] - (y0 + y1) / 2) / (fh / 2)
        if ox < -TURN_X:
            self.direction = "LEFT"
        elif ox > TURN_X:
            self.direction = "RIGHT"
        elif oy < LOOK_UP_Y:
            self.direction = "UP"
        elif oy > LOOK_DOWN_Y:
            self.direction = "DOWN"
        else:
            self.direction = "CENTER"
        self.away_since = (self.away_since or now) if self.direction != "CENTER" else None

        motion = 0.0 if self.prev_nose is None else float(np.linalg.norm(nose - self.prev_nose)) / fw
        self.prev_nose = nose

        targets = {
            "SMILE": (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2,
            "BROW RAISE": bs.get("browInnerUp", 0),
            "JAW OPEN": bs.get("jawOpen", 0),
            "EYE OPENNESS": 1 - blink,
            "HEAD MOTION": min(1.0, motion * 12),
        }
        for k, v in targets.items():
            self.values[k] += 0.3 * (v - self.values[k])

        closed_secs = now - self.closed_since if self.eyes_closed else 0.0
        away_secs = now - self.away_since if self.away_since else 0.0
        if closed_secs > 1.2:
            self.state = "DROWSY"
        elif away_secs > 2.0:
            self.state = "DISTRACTED"
        elif self.values["HEAD MOTION"] > 0.5:
            self.state = "RESTLESS"
        elif self.values["SMILE"] > 0.5:
            self.state = "ENGAGED"
        else:
            self.state = "FOCUSED"

        alert = max(closed_secs / 2.5, away_secs / 5.0, self.values["HEAD MOTION"],
                    (self.blink_rate - 20) / 20)
        self.alert += 0.08 * (float(np.clip(alert, 0, 1)) - self.alert)

        if self.frame_n % 3 == 0:
            self.thumb = self.face_thumb(gray)

    def face_thumb(self, gray):
        x0, y0, x1, y1 = self.bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_h = (y1 - y0) * 0.65
        half_w = half_h * THUMB_W / THUMB_H
        h, w = gray.shape
        a, b = int(max(0, cx - half_w)), int(min(w, cx + half_w))
        c, d = int(max(0, cy - half_h)), int(min(h, cy + half_h))
        if b - a < 8 or d - c < 8:
            return self.thumb
        crop = cv2.resize(gray[c:d, a:b], (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        return halftone(cv2.equalizeHist(crop), cell=5)

    def draw_face(self, img, now):
        if self.face_pts is None:
            return
        for i in range(0, len(self.face_pts), 4):
            cv2.circle(img, self.face_pts[i], 1, GREEN, -1)
        x0, y0, x1, y1 = self.bbox
        corner_brackets(img, x0 - 12, y0 - 12, x1 + 12, y1 + 12, STATE_COLORS[self.state])
        text(img, f"SUBJECT 01 // {self.state} // HEAD {self.direction}", (x0 - 12, y0 - 20),
             STATE_COLORS[self.state], 0.9)

    def draw_panel(self, img, now):
        x0 = PANEL_X
        y = panel_start(img, "BIOMETRIC READ // FACE ANALYSIS", 372)

        if self.thumb is not None:
            img[y:y + THUMB_H, x0:x0 + THUMB_W] = self.thumb
        cv2.rectangle(img, (x0 - 1, y - 1), (x0 + THUMB_W, y + THUMB_H), GREEN, 1)

        tracked = int(now - self.started) if self.started else 0
        rows = [
            (f"STATE: {self.state}", STATE_COLORS[self.state]),
            (f"BLINK RATE: {self.blink_rate:4.1f} /min", AMBER if self.blink_rate > 25 else GREEN),
            (f"EYES: {'CLOSED' if self.eyes_closed else 'OPEN'}", WHITE),
            (f"HEAD: {self.direction}", GREEN),
            (f"TRACKING: {tracked // 60:02d}:{tracked % 60:02d}", GREEN),
        ]
        tx = x0 + THUMB_W + 12
        for i, (s, color) in enumerate(rows):
            text(img, s, (tx, y + 14 + i * 18), color, 0.85)
        y += THUMB_H + 22

        text(img, "// FACIAL SIGNALS", (x0, y), GREEN, 0.9)
        y += 10
        for k in SIGNALS:
            value = self.values[k]
            text(img, k, (x0, y + 9), WHITE, 0.8)
            bar(img, x0 + 130, y + 1, 150, value)
            text(img, f"{int(value * 100):3d}%", (x0 + 290, y + 9), GREEN, 0.8)
            y += 18
        y += 18
        draw_level_meter(img, x0, y, self.alert, "ALERT LEVEL", now, PANEL_W)
