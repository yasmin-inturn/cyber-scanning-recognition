"""Modo SCAN: lê texto (OCR da Apple, offline) e QR codes dentro da área selecionada."""
import os
import subprocess
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import cv2

from hud import (AMBER, GREEN, PANEL_W, PANEL_X, RED, WHITE, ascii_only, panel_start,
                 quad_size, text, warp_quad, wrap_text)

MAX_LINES = 18
QR_EVERY_N_FRAMES = 6
QR_MEMORY = 1.5  # segundos que um QR continua "na mira" depois de sumir


def copy_to_clipboard(s):
    env = {**os.environ, "LANG": "en_US.UTF-8"}  # sem isso o pbcopy estraga os acentos
    subprocess.run(["pbcopy"], input=s.encode("utf-8"), env=env, check=False)


def unmirrored_crop(frame_bgr, quad, scale=2.0, max_side=2000):
    """Recorte reto e em alta da área. Desfaz o espelho da tela, senão o texto sai ao contrário."""
    w, h = quad_size(quad)
    k = min(scale, max_side / max(w, h))
    crop = warp_quad(frame_bgr, quad, max(int(w * k), 8), max(int(h * k), 8))
    return cv2.flip(crop, 1)


def run_ocr(img_bgr):
    from ocrmac import ocrmac
    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    results = ocrmac.OCR(pil, recognition_level="accurate", language_preference=["pt-BR", "en-US"]).recognize()
    # A caixa vem como (x, y, w, h) normalizada com origem embaixo: ordena de cima pra baixo
    results.sort(key=lambda r: (-round(r[2][1] + r[2][3], 2), r[2][0]))
    return "\n".join(r[0] for r in results)


class TextScanner:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)  # OCR roda em paralelo, o vídeo não trava
        self.qr_detector = cv2.QRCodeDetector()
        self.future = None
        self.text = ""
        self.status = "AWAITING TARGET"
        self.status_color = GREEN
        self.qr = None
        self.qr_seen = -10.0
        self.frame_n = 0

    def check_qr(self, frame_bgr, quad, now):
        self.frame_n += 1
        if self.frame_n % QR_EVERY_N_FRAMES:
            return
        gray = cv2.cvtColor(unmirrored_crop(frame_bgr, quad, scale=1.0), cv2.COLOR_BGR2GRAY)
        try:
            data, _, _ = self.qr_detector.detectAndDecode(gray)
        except cv2.error:
            data = ""
        if data:
            self.qr, self.qr_seen = data, now

    def qr_visible(self, now):
        return self.qr is not None and now - self.qr_seen < QR_MEMORY

    def action(self, frame_bgr, quad, now):
        """Gesto de ação: abre/copia o QR na mira, ou lê o texto da área. Retorna o aviso pra tela."""
        if self.qr_visible(now):
            copy_to_clipboard(self.qr)
            if self.qr.startswith(("http://", "https://")):
                webbrowser.open(self.qr)
                return "QR OPENED + COPIED"
            return "QR COPIED TO CLIPBOARD"
        if self.future is not None:
            return "STILL READING..."
        self.future = self.executor.submit(run_ocr, unmirrored_crop(frame_bgr, quad))
        self.status, self.status_color = "READING...", WHITE
        return "READING TEXT..."

    def poll(self):
        """Confere se o OCR terminou. Retorna um aviso quando termina, senão None."""
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        try:
            result = future.result().strip()
        except Exception as e:  # noqa: BLE001 - mostra qualquer erro do OCR na tela
            self.status, self.status_color = f"OCR ERROR: {e}"[:40], RED
            return "OCR ERROR"
        if not result:
            self.status, self.status_color = "NO TEXT FOUND", AMBER
            return "NO TEXT FOUND"
        self.text = result
        copy_to_clipboard(result)
        n = len(result.split())
        self.status, self.status_color = f"{n} WORDS // COPIED", GREEN
        return "TEXT COPIED TO CLIPBOARD"

    def draw_panel(self, img, now):
        lines = wrap_text(ascii_only(self.text), 44)
        if len(lines) > MAX_LINES:
            lines = lines[:MAX_LINES - 1] + [f"... (+{len(lines) - MAX_LINES + 1} lines)"]
        if not lines:
            lines = ["pinch THUMB + MIDDLE over a text", "or a QR code to read it."]
        height = 118 + 15 * len(lines)

        x0 = PANEL_X
        y = panel_start(img, "TEXT SCAN // OCR + QR", height)
        reading = self.future is not None
        status_color = WHITE if reading and int(now * 4) % 2 else self.status_color
        text(img, f"STATUS: {self.status}", (x0, y + 12), status_color, 0.9)
        if self.qr_visible(now):
            qr = self.qr if len(self.qr) < 38 else self.qr[:35] + "..."
            text(img, f"QR: {qr}", (x0, y + 30), AMBER, 0.9)
        else:
            text(img, "QR: none in target", (x0, y + 30), GREEN, 0.9)
        y += 52

        text(img, "// EXTRACTED TEXT", (x0, y), GREEN, 0.9)
        cv2.line(img, (x0, y + 6), (x0 + PANEL_W, y + 6), GREEN, 1)
        y += 24
        for line in lines:
            text(img, line, (x0, y), WHITE if self.text else GREEN, 0.8)
            y += 15

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
