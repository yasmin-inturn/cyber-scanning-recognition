"""
JARVIS // surveillance scanner

Webcam em preto e branco com HUD verde de câmera de segurança, controlada pelas mãos,
com leitura facial real (piscadas, expressões, direção da cabeça, nível de alerta).

Gestos (ao vivo):
  2 mãos  -> polegar e indicador de cada mão seguram os 4 cantos do scanner
  1 mão   -> polegar e indicador viram cantos opostos de um retângulo
  polegar + médio (segure)   -> zera os contadores da leitura facial
  polegar + anelar (segure)  -> REC: registro de câmera de segurança da área (clipe de 4 s)
  palma aberta parada (segure) -> abre o arquivo de evidências (ver evidence_viewer.py)

As evidências ficam só na memória: somem quando o programa fecha (nada é salvo no disco).

Teclas: G abre/fecha evidências | R grava a tela inteira
        ESC volta (ou sai, se estiver ao vivo) | Q sai

(text_scan.py e mouse_control.py ficaram guardados pra virar modos no futuro.)
"""
import time
from pathlib import Path

import cv2
import numpy as np

from evidence_viewer import EvidenceViewer
from face import BioReader, FaceTracker
from hands import (MIDDLE_TIP, RING_TIP, THUMB_TIP, HandTracker, Hold, PalmStill, Pinch, Spring,
                   corner_targets, quad_order)
from hud import (AMBER, GREEN, RED, WHITE, Toasts, build_film_mask, draw_hand, draw_hud,
                 draw_scanner, progress_ring, surveillance_filter)
from recorder import CCTVRecorder

HERE = Path(__file__).resolve().parent
HAND_MODEL = HERE / "hand_landmarker.task"
FACE_MODEL = HERE / "face_landmarker.task"
WINDOW = "JARVIS // surveillance scanner"
MODEL_URLS = {
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
HINT = "HOLD THUMB+MIDDLE: RESET BIO  //  HOLD THUMB+RING: REC AREA  //  HOLD OPEN PALM: EVIDENCE"


def require_models():
    missing = [p for p in MODEL_URLS if not p.exists()]
    if missing:
        cmds = "\n".join(f"  curl -L -o {p.name} {MODEL_URLS[p]}" for p in missing)
        raise SystemExit(f"Modelo(s) não encontrado(s). Baixe na pasta do projeto com:\n{cmds}")


def full_frame_quad(w, h):
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def main():
    require_models()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(
            "Não consegui abrir a câmera. No macOS, libere o acesso em "
            "Ajustes do Sistema > Privacidade e Segurança > Câmera para o PyCharm/Terminal."
        )

    h, w = frame.shape[:2]
    mask = build_film_mask(w, h)
    noises = [np.random.normal(0, 9, (h, w)).astype(np.float32) for _ in range(6)]

    hand_tracker = HandTracker(HAND_MODEL)
    face_tracker = FaceTracker(FACE_MODEL)
    springs = [Spring() for _ in range(4)]
    reset_pinch, rec_pinch = Pinch(MIDDLE_TIP), Pinch(RING_TIP)
    reset_hold, rec_hold, open_hold = Hold(0.35), Hold(0.45), Hold(1.0)
    palm_still = PalmStill()
    rec_quad = None
    bio = BioReader()
    recorder = CCTVRecorder()
    viewer = EvidenceViewer(recorder)
    toasts = Toasts()

    key = -1
    flash_until = 0.0
    fps, prev = 0.0, time.time()
    frame_i = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # espelho: mexer a mão pra direita move pra direita
            now = time.time()

            hands = hand_tracker.detect(frame)
            gray = surveillance_filter(frame, mask, noises[frame_i % len(noises)])
            frame_i += 1
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if key == ord("g"):
                if viewer.active:
                    viewer.state = None
                else:
                    viewer.open()
            elif viewer.active and key != 255:
                msg = viewer.handle_key(key)
                if msg:
                    toasts.push(msg)

            if viewer.active:
                # ---- arquivo de evidências (o vivo fica escuro ao fundo)
                for s in springs:
                    s.reset()
                msg = viewer.update(hands, now, w, h)
                if msg:
                    toasts.push(msg)
                for pts in hands:
                    draw_hand(out, pts)
                viewer.draw(out, now)
                hint = viewer.hint() if viewer.active else HINT
            else:
                # ---- ao vivo
                quad = quad_vel = None
                if hands:
                    pos = np.array([s.update(p) for s, p in zip(springs, corner_targets(hands))])
                    vel = np.array([s.vel for s in springs])
                    order = quad_order(pos)
                    quad, quad_vel = pos[order], vel[order]
                else:
                    for s in springs:
                        s.reset()

                # Congela a área no começo do gesto: o polegar (que é um canto) mexe durante a pinça
                reset_on = reset_pinch.update(hands) and quad is not None
                rec_on = rec_pinch.update(hands) and quad is not None
                rec_quad = (quad.copy() if rec_quad is None else rec_quad) if rec_on else None

                rings = []
                progress, fired = reset_hold.update(reset_on, now)
                if progress > 0:
                    rings.append((reset_pinch.hand[THUMB_TIP], progress, GREEN))
                if fired:
                    flash_until = now + 0.15
                    bio.reset()
                    toasts.push("BIO COUNTERS RESET")

                progress, fired = rec_hold.update(rec_on, now)
                if progress > 0:
                    rings.append((rec_pinch.hand[THUMB_TIP], progress, RED))
                if fired:
                    if recorder.recording:
                        toasts.push("ALREADY RECORDING", AMBER)
                    else:
                        toasts.push(f"REC {recorder.start(frame, rec_quad, now)} // CAPTURING", RED)
                if key == ord("r") and not recorder.recording:
                    toasts.push(f"REC {recorder.start(frame, full_frame_quad(w, h), now)} // FULL FRAME", RED)

                still, palm_center = palm_still.update(hands, w)
                progress, fired = open_hold.update(still, now)
                if progress > 0:
                    rings.append((palm_center, progress, WHITE))
                if fired:
                    viewer.open()
                    toasts.push("EVIDENCE ARCHIVE")

                bio.update(face_tracker.detect(frame), now, gray)
                bio.draw_face(out, now)

                if quad is not None and not still:
                    draw_scanner(out, gray, quad, quad_vel, now, reset_on or rec_on, now < flash_until,
                                 f"target // {bio.state}")
                for pts in hands:
                    draw_hand(out, pts)
                for center, progress, color in rings:
                    progress_ring(out, center, progress, color)
                bio.draw_panel(out, now)
                recorder.draw(out, now)
                hint = HINT

            # O registro continua gravando mesmo com o arquivo de evidências aberto
            saved = recorder.feed(frame, now)
            if saved:
                toasts.push(f"EVIDENCE {saved} STORED")

            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1 / dt)
            draw_hud(out, now, fps, len(hands), hint)
            toasts.draw(out, now)

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or (key == 27 and not viewer.active):
                break
    finally:
        hand_tracker.close()
        face_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
