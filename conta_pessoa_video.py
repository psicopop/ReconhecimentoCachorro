import argparse
import time
from typing import Optional
import cv2
from ultralytics import YOLO

# ---------- Núcleo: contar pessoas em um frame ----------
def contar_pessoas_no_frame(frame, model: YOLO, conf: float = 0.5):
    """
    Recebe um frame BGR (OpenCV), roda YOLO e retorna:
      - frame anotado
      - quantidade de pessoas detectadas
    """
    # Executa inferência
    results = model(frame, conf=conf, verbose=False)[0]

    # Se não houver detecções, apenas escreva Pessoas: 0
    if results.boxes is None or len(results.boxes) == 0:
        annotated = frame.copy()
        cv2.putText(annotated, "Pessoas: 0", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return annotated, 0

    names = results.names            # dict {id: nome}, ex. 0:'person'
    boxes = results.boxes            # boxes.xyxy, boxes.conf, boxes.cls
    classes = boxes.cls.int().tolist()

    # Filtra classe 'person'
    idx_person = [i for i, c in enumerate(classes) if names.get(int(c)) == "person"]
    qtde = len(idx_person)

    annotated = frame.copy()
    for i in idx_person:
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        conf_i = float(boxes.conf[i])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"person {conf_i:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Título com contagem
    cv2.putText(annotated, f"Pessoas: {qtde}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return annotated, qtde

# ---------- Loop da câmera ----------
def contar_da_camera(
    fonte_camera: int = 0,
    modelo: str = "yolov8n.pt",
    conf: float = 0.5,
    largura: int = 1280,
    altura: int = 720,
    salvar_video_em: Optional[str] = None,
    fps_destino: Optional[float] = None
):
    """
    Lê da câmera (fonte_camera), roda YOLO e mostra janela com boxes e contagem.
    Aperte 'q' ou ESC para sair.
    Se salvar_video_em for informado, grava o vídeo anotado (MP4).
    """
    # Carrega modelo
    model = YOLO(modelo)

    # Abre câmera
    cap = cv2.VideoCapture(fonte_camera)  # em Windows, teste: cv2.VideoCapture(fonte_camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    # Ajusta resolução
    if largura and altura:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    # FPS fonte
    fps_origem = cap.get(cv2.CAP_PROP_FPS)
    if fps_origem is None or fps_origem <= 0:
        fps_origem = 30.0  # fallback

    # Configura gravação, se necessário
    writer = None
    if salvar_video_em:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_gravacao = fps_destino or fps_origem
        writer = cv2.VideoWriter(salvar_video_em, fourcc, fps_gravacao, (int(cap.get(3)), int(cap.get(4))))
        if not writer.isOpened():
            print("Aviso: não foi possível abrir o gravador de vídeo. Continuando sem salvar.")
            writer = None

    ultima_medida = time.time()
    frames_contados = 0
    fps_movel = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame não lido da câmera. Encerrando.")
                break

            # Inferência + anotação
            anotado, qtd = contar_pessoas_no_frame(frame, model, conf=conf)

            # FPS estimado (média móvel simples)
            frames_contados += 1
            agora = time.time()
            if agora - ultima_medida >= 1.0:
                fps_movel = frames_contados / (agora - ultima_medida)
                frames_contados = 0
                ultima_medida = agora

           

            # Overlay de FPS
            cv2.putText(anotado, f"FPS: {fps_movel:.1f}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Mostra
            cv2.imshow("YOLOv8 - Pessoas (pressione 'q' para sair)", anotado)

            # Salva quadro, se habilitado
            if writer is not None:
                writer.write(anotado)

            # Tecla para sair
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' ou ESC
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Contar pessoas da câmera com YOLOv8")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão=0)")
    parser.add_argument("--modelo", type=str, default="yolov8n.pt", help="Caminho do modelo YOLOv8")
    parser.add_argument("--conf", type=float, default=0.5, help="Confiança mínima")
    parser.add_argument("--largura", type=int, default=1280, help="Largura do frame")
    parser.add_argument("--altura", type=int, default=720, help="Altura do frame")
    parser.add_argument("--salvar", type=str, default=None, help="Caminho do MP4 de saída (opcional)")
    parser.add_argument("--fps", type=float, default=None, help="FPS do vídeo salvo (opcional)")
    args = parser.parse_args()

    contar_da_camera(
        fonte_camera=args.camera,
        modelo=args.modelo,
        conf=args.conf,
        largura=args.largura,
        altura=args.altura,
        salvar_video_em=args.salvar,
        fps_destino=args.fps
    )

if __name__ == "__main__":
    main()
