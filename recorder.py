"""Registro de câmera de segurança.

Congela a área selecionada (como uma câmera fixa) e guarda um clipe curto com a "marca" de
DVR: câmera, data, hora, dia da semana, REC e número da evidência.

Tudo fica só na memória enquanto o programa está aberto: nada é salvo no disco.
Os frames são guardados comprimidos em JPEG (~30 KB cada) pra não pesar.
"""
from datetime import datetime

import cv2
import numpy as np

from hud import MARGIN, RED, WHITE, darken, quad_size, tag, text, text_width

CLIP_SECONDS = 4.0
CLIP_FPS = 15
OUT_H = 480
MAX_RECORDS = 60  # as mais antigas são descartadas depois disso
JPEG_QUALITY = 85
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
THUMB_BOX = (160, 110)
OSD_FONT = cv2.FONT_HERSHEY_SIMPLEX


def osd(img, s, org, scale):
    """Texto de DVR: branco com sombra preta."""
    x, y = org
    cv2.putText(img, s, (x + 2, y + 2), OSD_FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), OSD_FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)


def cctv_look(crop_bgr, rng):
    """P&B meio lavado, levemente borrado e com chiado, como câmera de segurança barata."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=-10).astype(np.float32)
    gray += rng.normal(0, 7, gray.shape)
    gray[::2] *= 0.93  # entrelaçado
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def encode(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]


def decode(buf):
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class CCTVRecorder:
    def __init__(self, cam="CAM 01"):
        self.cam = cam
        self.count = 0
        self.records = []  # mais nova primeiro: {"id", "when", "fps", "still", "frames"}
        self.current = None
        self.quad = None
        self.thumb = None
        self.rng = np.random.default_rng()

    @property
    def recording(self):
        return self.current is not None

    def start(self, frame_bgr, quad, now):
        """Começa um registro da área `quad` (em coordenadas da tela). Retorna o número da evidência."""
        qw, qh = quad_size(quad)
        ow = int(np.clip(qw * OUT_H / qh, 240, 960)) // 2 * 2
        self.size = (ow, OUT_H)
        dst = np.float32([[0, 0], [ow - 1, 0], [ow - 1, OUT_H - 1], [0, OUT_H - 1]])
        self.matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        self.quad = quad.astype(np.int32)

        self.count += 1
        self.start_t = self.next_frame = now
        self.frame_n = 0
        still = self.render(frame_bgr)
        self.current = {"id": f"#{self.count:04d}", "when": datetime.now(), "fps": CLIP_FPS,
                        "still": encode(still), "frames": []}
        k = min(THUMB_BOX[0] / ow, THUMB_BOX[1] / OUT_H)
        self.thumb = cv2.resize(still, (int(ow * k), int(OUT_H * k)))
        return self.current["id"]

    def render(self, frame_bgr):
        # Desfaz o espelho da tela: o registro sai como uma câmera de verdade veria
        crop = cv2.flip(cv2.warpPerspective(frame_bgr, self.matrix, self.size), 1)
        img = cctv_look(crop, self.rng)
        ow, oh = self.size
        scale = float(np.clip(ow / 900, 0.45, 0.8))
        line = int(36 * scale)
        dt = datetime.now()
        osd(img, self.cam, (10, 10 + line), scale)
        osd(img, f"{dt:%d-%m-%Y} {WEEKDAYS[dt.weekday()]} {dt:%H:%M:%S}", (10, 18 + 2 * line), scale)
        cv2.circle(img, (18, oh - 10 - line // 2), max(4, line // 3), (0, 0, 255), -1, cv2.LINE_AA)
        osd(img, "REC", (30, oh - 10), scale)
        evidence = f"EVIDENCE #{self.count:04d}  F{self.frame_n:04d}"
        (tw, _), _ = cv2.getTextSize(evidence, OSD_FONT, scale, 2)
        osd(img, evidence, (ow - tw - 10, oh - 10), scale)
        return img

    def feed(self, frame_bgr, now):
        """Grava o frame atual se estiver gravando. Retorna o número da evidência quando o clipe termina."""
        if not self.recording:
            return None
        if now >= self.next_frame:
            self.current["frames"].append(encode(self.render(frame_bgr)))
            self.frame_n += 1
            self.next_frame = max(self.next_frame + 1 / CLIP_FPS, now - 0.5 / CLIP_FPS)
        if now - self.start_t >= CLIP_SECONDS:
            return self.finish()
        return None

    def finish(self):
        if self.current is None:
            return None
        record, self.current = self.current, None
        self.records.insert(0, record)
        del self.records[MAX_RECORDS:]
        return record["id"]

    def draw(self, img, now):
        h, w = img.shape[:2]
        if self.recording:
            cv2.polylines(img, [self.quad], True, RED, 2, cv2.LINE_AA)
            secs = now - self.start_t
            label = f"REC  00:{int(secs):02d} / 00:{int(CLIP_SECONDS):02d}"
            x = (w - text_width(label, 1.3)) // 2
            if int(now * 2) % 2 == 0:
                cv2.circle(img, (x - 14, MARGIN + 52), 7, RED, -1, cv2.LINE_AA)
            text(img, label, (x, MARGIN + 58), RED, 1.3)

        if self.thumb is not None:
            th, tw = self.thumb.shape[:2]
            x0, y0 = MARGIN + 16, h - MARGIN - 40 - th
            darken(img, x0 - 6, y0 - 30, x0 + max(tw, 170) + 6, y0 + th + 6)
            last = self.current or self.records[0]
            tag(img, f"EVIDENCE LOG {last['id']}  {last['when']:%H:%M:%S}", x0, y0 - 26, scale=0.85)
            img[y0:y0 + th, x0:x0 + tw] = self.thumb
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), RED if self.recording else WHITE, 1)
