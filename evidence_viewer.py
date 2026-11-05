"""Arquivo de evidências: galeria das gravações do REC e player do clipe, tudo controlado pelas mãos.

Galeria:  aponte com o indicador | segure pinça polegar+indicador pra abrir
          palma aberta deslizando: troca de página | punho fechado (segure): fecha
Player:   pinça polegar+indicador e arraste: avança/volta no clipe
          segure pinça polegar+médio: play/pause | punho fechado (segure): volta pra galeria
"""
import cv2
import numpy as np

from hands import INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, Hold, Pinch, SwipeDetector, is_fist
from hud import AMBER, GREEN, MARGIN, WHITE, corner_brackets, darken, progress_ring, tag, text, text_width
from recorder import decode

COLS, ROWS = 4, 2
PER_PAGE = COLS * ROWS
GAP = 18

HINTS = {
    "GALLERY": "POINT: SELECT  //  HOLD PINCH THUMB+INDEX: OPEN  //  SWIPE PALM: PAGE  //  HOLD FIST: CLOSE",
    "PLAYER": "PINCH THUMB+INDEX + MOVE: SCRUB  //  HOLD PINCH THUMB+MIDDLE: PLAY/PAUSE  //  HOLD FIST: BACK",
}


def fit_into(img, box_w, box_h):
    h, w = img.shape[:2]
    k = min(box_w / w, box_h / h)
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)


def paste_centered(dst, img, x, y, box_w, box_h):
    ih, iw = img.shape[:2]
    ox, oy = x + (box_w - iw) // 2, y + (box_h - ih) // 2
    dst[oy:oy + ih, ox:ox + iw] = img


def load_clip(record):
    """Descomprime os frames do clipe pra tocar (se o clipe ainda não tiver frames, usa a foto)."""
    frames = [decode(buf) for buf in record["frames"]] or [decode(record["still"])]
    return frames, float(record["fps"])


class EvidenceViewer:
    def __init__(self, recorder):
        self.recorder = recorder  # as evidências vivem na memória do recorder
        self.state = None  # None (fechado), "GALLERY" ou "PLAYER"
        self.records = []
        self.thumbs = {}
        self.page = 0
        self.pointer = None
        self.frozen_pointer = None
        self.hover = None
        self.rings = []
        self.select_pinch, self.select_hold = Pinch(INDEX_TIP), Hold(0.3)
        self.pause_pinch, self.pause_hold = Pinch(MIDDLE_TIP), Hold(0.35)
        self.back_hold = Hold(0.8)
        self.swipe = SwipeDetector()
        self.record = None
        self.frames = []
        self.fps = 15.0
        self.pos = 0.0  # posição no clipe, em segundos
        self.playing = False
        self.scrubbing = False
        self.was_playing = False
        self.last_now = None

    # ---------------------------------------------------------------- estado

    @property
    def active(self):
        return self.state is not None

    @property
    def pages(self):
        return max(1, -(-len(self.records) // PER_PAGE))

    def hint(self):
        return HINTS[self.state]

    def open(self):
        self.records = list(self.recorder.records)
        self.thumbs.clear()
        self.page = 0
        self.state = "GALLERY"

    def back(self):
        if self.state == "PLAYER":
            self.state = "GALLERY"
            self.frames = []
            return "BACK TO ARCHIVE"
        self.state = None
        return "ARCHIVE CLOSED"

    def play(self, record):
        self.record = record
        self.frames, self.fps = load_clip(record)
        self.pos = 0.0
        self.playing = True
        self.scrubbing = False
        self.state = "PLAYER"

    def duration(self):
        return len(self.frames) / self.fps if self.frames else 0.0

    def toggle_play(self):
        if not self.playing and self.pos >= self.duration() - 1e-3:
            self.pos = 0.0  # no fim, play recomeça do início
        self.playing = not self.playing
        return "PLAY" if self.playing else "PAUSED"

    def handle_key(self, key):
        """ESC volta | ESPAÇO play/pause | A/D troca de página."""
        if key == 27:
            return self.back()
        if self.state == "PLAYER" and key == ord(" "):
            return self.toggle_play()
        if self.state == "GALLERY" and key in (ord("a"), ord("d")):
            self.page = int(np.clip(self.page + (1 if key == ord("d") else -1), 0, self.pages - 1))
        return None

    # ---------------------------------------------------------------- geometria

    def tile_rects(self, w, h):
        x0, x1 = MARGIN + 40, w - MARGIN - 40
        y0, y1 = MARGIN + 70, h - MARGIN - 60
        tw = (x1 - x0 - GAP * (COLS - 1)) // COLS
        th = (y1 - y0 - GAP * (ROWS - 1)) // ROWS
        start = self.page * PER_PAGE
        rects = []
        for i, rec in enumerate(self.records[start:start + PER_PAGE]):
            r, c = divmod(i, COLS)
            rects.append((rec, x0 + c * (tw + GAP), y0 + r * (th + GAP), tw, th))
        return rects

    def player_layout(self, w, h):
        """Retângulo do vídeo (x, y, w, h) e da linha do tempo (x0, y, x1)."""
        bx0, bx1 = MARGIN + 40, w - MARGIN - 40
        by0, by1 = MARGIN + 60, h - MARGIN - 130
        fh, fw = self.frames[0].shape[:2] if self.frames else (480, 640)
        k = min((bx1 - bx0) / fw, (by1 - by0) / fh)
        vw, vh = int(fw * k), int(fh * k)
        vx, vy = (w - vw) // 2, by0 + (by1 - by0 - vh) // 2
        return (vx, vy, vw, vh), (vx, vy + vh + 30, vx + vw)

    # ---------------------------------------------------------------- gestos

    def update(self, hands, now, w, h):
        """Processa os gestos do frame. Retorna um aviso pra tela ou None."""
        self.rings = []
        hand = hands[0] if hands else None
        self.pointer = hand[INDEX_TIP] if hand else None
        dt = 0.0 if self.last_now is None else now - self.last_now
        self.last_now = now

        progress, fired = self.back_hold.update(hand is not None and is_fist(hand), now)
        if progress > 0:
            self.rings.append((hand[MIDDLE_MCP], progress, AMBER))
        if fired:
            return self.back()

        if self.state == "GALLERY":
            return self.update_gallery(hands, now, w, h)
        return self.update_player(hands, now, dt, w, h)

    def update_gallery(self, hands, now, w, h):
        direction = self.swipe.update(hands, w, now)
        if direction:
            # Mão indo pra esquerda = próxima página (como passar folhas)
            page = int(np.clip(self.page - direction, 0, self.pages - 1))
            if page != self.page:
                self.page = page
                return f"PAGE {self.page + 1}/{self.pages}"

        # Congela a mira no começo da pinça: fechar os dedos mexe a ponta do indicador
        pinching = self.select_pinch.update(hands)
        if not pinching:
            self.frozen_pointer = None
        elif self.frozen_pointer is None:
            self.frozen_pointer = self.select_pinch.hand[INDEX_TIP]
        point = self.frozen_pointer or self.pointer

        self.hover = None
        if point is not None:
            for rec, x, y, tw, th in self.tile_rects(w, h):
                if x <= point[0] <= x + tw and y <= point[1] <= y + th:
                    self.hover = rec

        progress, fired = self.select_hold.update(pinching and self.hover is not None, now)
        if progress > 0:
            self.rings.append((self.select_pinch.hand[THUMB_TIP], progress, GREEN))
        if fired:
            self.play(self.hover)
            return f"PLAYBACK EVIDENCE {self.record['id']}"
        return None

    def update_player(self, hands, now, dt, w, h):
        duration = self.duration()

        # Arrastar: pinça polegar+indicador e a posição da mão vira a posição no clipe
        scrub = self.select_pinch.update(hands)
        if scrub and not self.scrubbing:
            self.was_playing, self.playing = self.playing, False
        elif self.scrubbing and not scrub:
            self.playing = self.was_playing and self.pos < duration
        self.scrubbing = scrub

        if scrub:
            _, (tx0, _, tx1) = self.player_layout(w, h)
            x = self.select_pinch.hand[INDEX_TIP][0]
            self.pos = float(np.clip((x - tx0) / max(tx1 - tx0, 1), 0, 1)) * duration
        elif self.playing:
            self.pos += dt
            if self.pos >= duration:
                self.pos, self.playing = duration, False

        progress, fired = self.pause_hold.update(self.pause_pinch.update(hands), now)
        if progress > 0:
            self.rings.append((self.pause_pinch.hand[THUMB_TIP], progress, GREEN))
        if fired:
            return self.toggle_play()
        return None

    # ---------------------------------------------------------------- desenho

    def draw(self, img, now):
        h, w = img.shape[:2]
        darken(img, 0, 0, w, h, 0.18)  # o vivo continua ao fundo, bem escuro
        if self.state == "GALLERY":
            self.draw_gallery(img, w, h)
        else:
            self.draw_player(img, now, w, h)

        if self.pointer is not None:
            x, y = self.pointer
            cv2.circle(img, (x, y), 10, WHITE, 1, cv2.LINE_AA)
            cv2.line(img, (x - 16, y), (x + 16, y), WHITE, 1)
            cv2.line(img, (x, y - 16), (x, y + 16), WHITE, 1)
        for center, progress, color in self.rings:
            progress_ring(img, center, progress, color)

    def draw_gallery(self, img, w, h):
        title = f"EVIDENCE ARCHIVE // {len(self.records)} RECORDS"
        tag(img, title, MARGIN + 16, MARGIN + 16)
        text(img, f"PAGE {self.page + 1}/{self.pages}", (MARGIN + 40 + text_width(title), MARGIN + 32))

        if not self.records:
            msg = "NO EVIDENCE IN THIS SESSION // HOLD THUMB+RING ON LIVE VIEW TO RECORD"
            text(img, msg, ((w - text_width(msg, 1.3)) // 2, h // 2), GREEN, 1.3)
            return

        for rec, x, y, tw, th in self.tile_rects(w, h):
            box_h = th - 40
            if rec["id"] not in self.thumbs:
                self.thumbs[rec["id"]] = fit_into(decode(rec["still"]), tw - 8, box_h - 8)

            cv2.rectangle(img, (x, y), (x + tw, y + box_h), (10, 10, 10), -1)
            paste_centered(img, self.thumbs[rec["id"]], x + 4, y + 4, tw - 8, box_h - 8)
            hovered = rec is self.hover
            color = AMBER if hovered else GREEN
            cv2.rectangle(img, (x, y), (x + tw, y + box_h), color, 2 if hovered else 1)
            if hovered:
                corner_brackets(img, x - 6, y - 6, x + tw + 6, y + box_h + 6, AMBER)

            text(img, f"EVIDENCE {rec['id']}", (x, y + box_h + 16), color, 0.95)
            text(img, f"{rec['when']:%d-%m-%Y  %H:%M:%S}", (x, y + box_h + 32), WHITE, 0.8)
            secs = f"{len(rec['frames']) / rec['fps']:.1f}s"
            text(img, secs, (x + tw - text_width(secs, 0.8), y + box_h + 32), GREEN, 0.8)

    def draw_player(self, img, now, w, h):
        rec = self.record
        (vx, vy, vw, vh), (tx0, ty, tx1) = self.player_layout(w, h)
        title = f"EVIDENCE {rec['id']} // PLAYBACK"
        tag(img, title, MARGIN + 16, MARGIN + 16)
        text(img, f"REC {rec['when']:%d-%m-%Y  %H:%M:%S}", (MARGIN + 40 + text_width(title), MARGIN + 32), WHITE)

        n = len(self.frames)
        idx = 0
        if n:
            idx = min(int(self.pos * self.fps), n - 1)
            img[vy:vy + vh, vx:vx + vw] = cv2.resize(self.frames[idx], (vw, vh), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(img, (vx - 1, vy - 1), (vx + vw, vy + vh), GREEN, 1)
        corner_brackets(img, vx - 10, vy - 10, vx + vw + 10, vy + vh + 10)

        # Linha do tempo com marcas de segundo e a agulha
        duration = self.duration()
        frac = self.pos / duration if duration else 0.0
        px = tx0 + int(frac * (tx1 - tx0))
        cv2.rectangle(img, (tx0, ty), (tx1, ty + 10), GREEN, 1)
        cv2.rectangle(img, (tx0, ty), (px, ty + 10), GREEN, -1)
        if duration:
            for s in range(int(duration) + 1):
                sx = tx0 + int(s / duration * (tx1 - tx0))
                cv2.line(img, (sx, ty + 12), (sx, ty + 18), GREEN, 1)
                text(img, f"{s}s", (sx - 6, ty + 32), GREEN, 0.75)
        cv2.line(img, (px, ty - 10), (px, ty + 20), WHITE, 2)
        cv2.circle(img, (px, ty - 10), 4, WHITE, -1, cv2.LINE_AA)

        if self.scrubbing:
            status, color = "<< SCRUB >>", AMBER
        elif self.playing:
            status, color = "> PLAY", GREEN
        elif self.pos >= duration:
            status, color = "[] END", WHITE
        else:
            status, color = "|| PAUSED", WHITE
        if not (self.playing and int(now * 2) % 2):
            text(img, status, (tx0, ty + 54), color, 1.2)
        timecode = f"{self.pos:05.2f}s / {duration:05.2f}s    F{idx + 1:04d}/{max(n, 1):04d}"
        text(img, timecode, (tx1 - text_width(timecode, 1.1), ty + 54), WHITE, 1.1)
