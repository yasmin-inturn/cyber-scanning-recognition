# JARVIS // surveillance scanner

Webcam em preto e branco com HUD verde de câmera de segurança, controlada pelas mãos, com leitura
facial ao vivo e registro de evidências em vídeo. Feito com MediaPipe e OpenCV.

- **Leitura facial:** piscadas, expressões, direção da cabeça e nível de alerta. Mede o que o rosto
  faz, não o que a pessoa sente, e não identifica ninguém.
- **Gravação:** clipes de 4 segundos da área selecionada, com visual de câmera de segurança.
- **Arquivo de evidências:** galeria e player controlados por gestos.

As evidências ficam só na memória e somem quando o programa fecha. Nada é salvo no disco.

A primeira versão, apenas visual, está em um projeto separado.

## Como rodar

Testado com Python 3.14 no macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
python jarvis_scanner.py
```

No macOS, libere a câmera para o PyCharm ou o Terminal em
Ajustes do Sistema > Privacidade e Segurança > Câmera.

## Controles

### Ao vivo

| Gesto | Ação |
| --- | --- |
| 2 mãos | Polegar e indicador de cada mão seguram os 4 cantos do scanner |
| 1 mão | Polegar e indicador viram cantos opostos de um retângulo |
| Segurar polegar + médio | Zera os contadores da leitura facial |
| Segurar polegar + anelar | Grava a área selecionada |
| Segurar a palma aberta parada | Abre o arquivo de evidências |

### Arquivo de evidências

| Gesto | Ação |
| --- | --- |
| Apontar com o indicador | Seleciona um clipe na galeria |
| Segurar polegar + indicador | Abre o clipe |
| Palma aberta deslizando | Troca de página |
| Polegar + indicador e arrastar | Avança ou volta no clipe |
| Segurar polegar + médio | Play/pause |
| Segurar o punho fechado | Volta para a galeria ou fecha o arquivo |

### Teclado

| Tecla | Ação |
| --- | --- |
| `G` | Abre ou fecha o arquivo de evidências |
| `R` | Grava a tela inteira |
| `Esc` | Volta ou, se estiver ao vivo, sai |
| `Q` | Sai |

## Arquivos

| Arquivo | Conteúdo |
| --- | --- |
| `jarvis_scanner.py` | Programa principal |
| `hands.py` | Rastreamento das mãos e gestos |
| `face.py` | Leitura facial |
| `hud.py` | Visual e efeitos |
| `recorder.py` | Gravação dos clipes |
| `evidence_viewer.py` | Galeria e player |
| `text_scan.py`, `mouse_control.py` | Módulos guardados para modos futuros, ainda não ligados ao programa |
