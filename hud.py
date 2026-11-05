"""Visual do HUD: cores, textos, efeitos de câmera de vigilância e o scanner fluido."""
import time
import unicodedata
from datetime import datetime

import cv2
import numpy as np

from hands import FINGER_TIPS, HAND_CONNECTIONS, WRIST

# Cores em BGR (o OpenCV usa BGR, não RGB)
GREEN = (40, 255, 60)
DARK_GREEN = (12, 50, 14)
AMBER = (0, 190, 255)
WHITE = (235, 235, 235)
BLACK = (0, 0, 0)
RED = (40, 40, 230)
FONT = cv2.FONT_HERSHEY_PLAIN

MARGIN = 28
PANEL_X, PANEL_Y, PANEL_W = MARGIN + 16, MARGIN + 16, 340
THUMB_W, THUMB_H = 110, 130

LEVELS = [
    (0.2, "LOW", GREEN),
    (0.4, "GUARDED", GREEN),
    (0.6, "ELEVATED", AMBER),
    (0.8, "HIGH", RED),
    (1.01, "CRITICAL", RED),
]


# ---------------------------------------------------------------- efeitos

def build_film_mask(w, h):
    """Vinheta + scanlines pré-calculadas (multiplicam o frame em P&B)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vignette = np.clip(1.15 - 0.45 * d2, 0.35, 1.0)
    scanlines = np.where(np.arange(h) % 3 == 0, 0.82, 1.0).astype(np.float32)[:, None]
    return vignette * scanlines


def surveillance_filter(frame, mask, noise):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = (gray - 128) * 1.35 + 128  # mais contraste
    gray = gray * mask + noise         # vinheta, scanlines e granulado
    return np.clip(gray, 0, 255).astype(np.uint8)


def make_lut(dark, light):
    t = np.linspace(0, 1, 256)[:, None]
    return (np.array(dark) * (1 - t) + np.array(light) * t).astype(np.uint8)


GREEN_LUT = make_lut(DARK_GREEN, GREEN)


def green_tint(gray):
    return GREEN_LUT[gray]


def halftone(gray, cell=6):
    """Retícula verde estilo pôster: pontos escuros maiores onde a imagem é mais escura."""
    h, w = gray.shape
    out = np.full((h, w, 3), GREEN, np.uint8)
    small = cv2.resize(gray, (w // cell, h // cell), interpolation=cv2.INTER_AREA)
    r_max = cell * 0.72
    for j in range(small.shape[0]):
        for i in range(small.shape[1]):
            r = (1 - small[j, i] / 255.0) * r_max
            if r > 0.6:
                center = (i * cell + cell // 2, j * cell + cell // 2)
                cv2.circle(out, center, int(round(r)), DARK_GREEN, -1, cv2.LINE_AA)
    return out


def warp_quad(img, quad, w, h):
    """'Desentorta' o formato pra um retângulo, como uma foto tirada de frente."""
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, matrix, (w, h))


def quad_size(quad):
    """Largura e altura médias do quadrilátero (TL, TR, BR, BL)."""
    q = quad.astype(np.float32)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return max(float(w), 1.0), max(float(h), 1.0)


# ---------------------------------------------------------------- texto e peças

def ascii_only(s):
    """O putText do OpenCV não desenha acentos: 'ação' vira 'acao' só na tela."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def text(img, s, org, color=GREEN, scale=1.0):
    cv2.putText(img, ascii_only(s), org, FONT, scale, color, 1, cv2.LINE_AA)


def text_width(s, scale=1.0):
    return cv2.getTextSize(ascii_only(s), FONT, scale, 1)[0][0]


def tag(img, s, x, y, fg=BLACK, bg=GREEN, scale=1.0, pad=4, width=None):
    """Etiqueta com fundo sólido (tipo 'ID: P-2204-02'). Retorna o y da próxima linha."""
    (tw, th), base = cv2.getTextSize(ascii_only(s), FONT, scale, 1)
    w = width or tw + 2 * pad
    cv2.rectangle(img, (x, y), (x + w, y + th + base + 2 * pad), bg, -1)
    text(img, s, (x + pad, y + th + pad), fg, scale)
    return y + th + base + 2 * pad + 4


def wrap_text(s, width):
    lines = []
    for para in s.splitlines():
        while len(para) > width:
            cut = para.rfind(" ", 0, width)
            cut = width if cut <= 0 else cut
            lines.append(para[:cut])
            para = para[cut:].lstrip()
        lines.append(para)
    return lines


def bar(img, x, y, w, value, color=GREEN, h=8):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    fill = int((w - 2) * float(np.clip(value, 0, 1)))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), color, -1)


def darken(img, x0, y0, x1, y1, factor=0.3):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * factor).astype(np.uint8)


def panel_start(img, title, height):
    """Fundo escuro + título do painel da esquerda. Retorna o y onde o conteúdo começa."""
    darken(img, PANEL_X - 6, PANEL_Y - 6, PANEL_X + PANEL_W + 6, PANEL_Y + height)
    return tag(img, title, PANEL_X, PANEL_Y, width=PANEL_W)


def dashed_rect(img, p1, p2, color, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, 1)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, 1)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, 1)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, 1)


def corner_brackets(img, x0, y0, x1, y1, color=GREEN, length=18, thickness=2):
    for (cx, cy), (dx, dy) in (((x0, y0), (1, 1)), ((x1, y0), (-1, 1)), ((x1, y1), (-1, -1)), ((x0, y1), (1, -1))):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness)


def progress_ring(img, center, progress, color=GREEN, radius=22):
    center = (int(center[0]), int(center[1]))
    cv2.circle(img, center, radius, DARK_GREEN, 2, cv2.LINE_AA)
    cv2.ellipse(img, center, (radius, radius), -90, 0, int(360 * progress), color, 3, cv2.LINE_AA)


def level_label(value):
    for limit, name, color in LEVELS:
        if value < limit:
            return name, color
    return LEVELS[-1][1:]


def draw_level_meter(img, x, y, value, title, now, width=PANEL_W):
    """Barra de 10 blocos LOW -> CRITICAL (verde, âmbar, vermelho)."""
    name, color = level_label(value)
    text(img, title, (x, y), WHITE, 0.9)
    text(img, f"{value * 100:5.1f} / 100", (x + 240, y), color, 0.9)
    y += 8
    filled = int(round(value * 10))
    for i in range(10):
        sx = x + i * 25
        cv2.rectangle(img, (sx, y), (sx + 20, y + 14), color, -1 if i < filled else 1)
    if not (name == "CRITICAL" and int(now * 4) % 2 == 0):
        tag(img, name, x + 255, y - 2, bg=color, scale=0.9, width=width - 255)
    return y + 24


class Toasts:
    """Avisos rápidos no centro inferior da tela."""

    def __init__(self):
        self.items = []

    def push(self, msg, color=GREEN, seconds=2.5):
        self.items.append((msg, color, time.time() + seconds))

    def draw(self, img, now):
        self.items = [it for it in self.items if it[2] > now][-3:]
        h, w = img.shape[:2]
        y = h - MARGIN - 64
        for msg, color, _ in reversed(self.items):
            tag(img, msg, (w - text_width(msg, 1.2)) // 2 - 6, y, bg=color, scale=1.2)
            y -= 30


# ---------------------------------------------------------------- mãos e scanner

def draw_hand(img, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], GREEN, 1, cv2.LINE_AA)
    for i in FINGER_TIPS:
        cv2.circle(img, pts[i], 3, GREEN, -1, cv2.LINE_AA)
    x, y = pts[WRIST]
    dashed_rect(img, (x - 16, y - 16), (x + 16, y + 16), GREEN)


def fluid_outline(quad, vels, t, samples=20):
    """Bordas curvas: cada lado entorta pra trás quando é arrastado e ondula de leve."""
    pts = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge)) + 1e-6
        normal = np.array([edge[1], -edge[0]], np.float32) / length
        drag = -float(np.dot((vels[i] + vels[(i + 1) % 4]) / 2, normal)) * 2.0
        wobble = np.sin(t * 2.4 + i * 1.7) * min(6.0, length * 0.03)
        bulge = np.clip(drag + wobble, -0.25 * length, 0.25 * length)
        ctrl = (a + b) / 2 + normal * bulge
        s = np.linspace(0, 1, samples, endpoint=False)[:, None]
        pts.append((1 - s) ** 2 * a + 2 * (1 - s) * s * ctrl + s ** 2 * b)
    return np.vstack(pts).astype(np.int32)


def draw_scanner(img, gray, quad, vels, t, locked, flash, label):
    h, w = gray.shape
    outline = fluid_outline(quad, vels, t)
    x, y, bw, bh = cv2.boundingRect(outline)
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return

    # Máscara suave (anti-aliasing) com o formato fluido, só na região do scanner
    local = outline - np.array([x0, y0], np.int32)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [local], 255, cv2.LINE_AA)
    alpha = mask[..., None].astype(np.float32) / 255.0

    roi = img[y0:y1, x0:x1]
    tinted = green_tint(gray[y0:y1, x0:x1]).astype(np.float32)
    if flash:
        tinted = tinted * 0.3 + 255 * 0.7
    roi[:] = (roi * (1 - alpha) + tinted * alpha).astype(np.uint8)

    # Brilho em volta da borda
    glow = roi.copy()
    cv2.polylines(glow, [local], True, GREEN, 8, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.25, roi, 0.75, 0, dst=roi)

    # Linha de varredura subindo e descendo, recortada pelo formato
    y_line = y + int(abs(((t * 0.9) % 2) - 1) * max(bh - 1, 0))
    if y0 <= y_line < y1:
        inside = mask[y_line - y0] > 128
        img[y_line, x0:x1][inside] = WHITE

    cv2.polylines(img, [outline], True, GREEN, 2 if locked else 1, cv2.LINE_AA)

    # Cantos "agarrados" pelos dedos
    for i, (cx, cy) in enumerate(quad.astype(int)):
        r = int(10 + 3 * np.sin(t * 5 + i * 1.3))
        cv2.circle(img, (cx, cy), r, GREEN, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, WHITE, -1, cv2.LINE_AA)
        text(img, f"{max(cx, 0):04d}.{max(cy, 0):04d}", (cx + 14, cy - 10), GREEN, 0.7)

    # Mira no centro
    mx, my = quad.mean(axis=0).astype(int)
    cv2.line(img, (mx - 8, my), (mx + 8, my), GREEN, 1)
    cv2.line(img, (mx, my - 8), (mx, my + 8), GREEN, 1)

    text(img, label, (x, min(y + bh + 20, h - 4)), GREEN, 0.9)


# ---------------------------------------------------------------- moldura

def draw_hud(img, t, fps, n_hands, hint, modes=None, mode_i=0):
    h, w = img.shape[:2]
    m = MARGIN
    cv2.rectangle(img, (m, m), (w - m, h - m), WHITE, 1)

    text(img, "CAM_01 // surveillance node active", (m, m - 9))
    stamp = datetime.now().strftime("%Y.%m.%d  %H:%M:%S")
    text(img, stamp, (w - m - text_width(stamp), m - 9))

    # Abas de modo no topo (só quando houver mais de um modo)
    labels = [f" {i + 1} {name} " for i, name in enumerate(modes or ())]
    total = sum(text_width(s) + 14 for s in labels)
    x = (w - total) // 2
    for i, s in enumerate(labels):
        tw = text_width(s) + 8
        if i == mode_i:
            tag(img, s, x, m + 8, width=tw)
        else:
            cv2.rectangle(img, (x, m + 8), (x + tw, m + 30), GREEN, 1)
            text(img, s, (x + 4, m + 25), GREEN)
        x += tw + 6

    # Régua inferior + dica de gestos do modo
    yb = h - m
    for x in range(m, w - m, 80):
        cv2.line(img, (x, yb), (x, yb - 8), WHITE, 1)
    cv2.circle(img, (w // 2, yb), 6, WHITE, 1, cv2.LINE_AA)
    text(img, hint, ((w - text_width(hint, 0.85)) // 2, yb - 16), WHITE, 0.85)

    text(img, "signal strength: 92%", (m, h - 9))
    mid = f"hands: {n_hands}  //  noise level: minimal"
    text(img, mid, ((w - text_width(mid)) // 2, h - 9))
    right = f"output filter: active  {fps:4.0f} fps"
    text(img, right, (w - m - text_width(right), h - 9))

    if int(t * 2) % 2 == 0:  # LIVE piscando
        cv2.circle(img, (w - m - 56, m + 22), 5, GREEN, -1, cv2.LINE_AA)
    text(img, "LIVE", (w - m - 44, m + 27), WHITE)
