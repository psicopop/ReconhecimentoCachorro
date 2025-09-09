import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def contar_animais(imagem_path: str, saida_path: str | None = None,
                   conf: float = 0.5, modelo: str = "yolov8n.pt", animal: str = "dog") -> int:
    """
    Conta um tipo específico de animal em uma imagem usando YOLOv8 e salva uma cópia anotada.

    """
    model = YOLO(modelo)

    # Executa a inferência
    results = model(imagem_path, conf=conf, verbose=False)[0]

    # Se não houver detecções
    if results.boxes is None or len(results.boxes) == 0:
        img = cv2.imread(imagem_path)
        if img is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem: {imagem_path}")
        # escreve contagem 0 e salva
        cv2.putText(img, f"{animal.capitalize()}: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        saida_path = saida_path or _default_out_path(imagem_path, animal)
        cv2.imwrite(saida_path, img)
        return 0

    names = results.names  # dict {id: nome}
    boxes = results.boxes  # Boxes com xyxy, conf, cls

    # Filtra apenas a classe desejada
    classes = boxes.cls.int().tolist()
    idx_animal = [i for i, c in enumerate(classes) if names[int(c)] == animal]
    qtde = len(idx_animal)

    # Desenha caixas apenas dos animais
    img = cv2.imread(imagem_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {imagem_path}")

    for i in idx_animal:
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        conf_i = float(boxes.conf[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{animal} {conf_i:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Título com contagem
    cv2.putText(img, f"{animal.capitalize()}: {qtde}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    saida_path = saida_path or _default_out_path(imagem_path, animal)
    cv2.imwrite(saida_path, img)

    return qtde

def _default_out_path(imagem_path: str, animal: str) -> str:
    p = Path(imagem_path)
    return str(p.with_name(f"{p.stem}_{animal}s{p.suffix}"))

def main():
    # Exemplo para contar cachorros
    qtde_cachorros = contar_animais('entrada.jpg', 'saida_cachorros.jpg', conf=0.6, animal='dog')
    print(f"Contagem de cachorros: {qtde_cachorros}")

    

if __name__ == "__main__":
    main()