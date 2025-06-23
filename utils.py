import tensorflow as tf
import numpy as np
from config import anchors_tf


def yolo_iou(box_wh, anchors):
    """
    Calcula a IoU entre uma bounding box (apenas largura e altura) e um conjunto de âncoras.
    
    Parâmetros:
    - box_wh: tensor/lista com [w, h] da bounding box (valores normalizados entre 0 e 1)
    - anchors: tensor (3, 2), cada linha é uma âncora [w, h], já normalizada entre 0 e 1

    Retorna:
    - ious: vetor de IoUs com cada âncora
    """
    # Converter entrada para tensores TensorFlow
    box_wh = tf.convert_to_tensor(box_wh, dtype=tf.float32)  # shape (2,)
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)  # shape (3, 2)

    # Calcular interseção
    inter_wh = tf.minimum(box_wh, anchors)  # shape (3, 2)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # shape (3,)

    # Calcular área da caixa e das âncoras
    box_area = box_wh[0] * box_wh[1]  # escalar
    anchor_area = anchors[..., 0] * anchors[..., 1]  # shape (3,)

    # Calcular união
    union_area = box_area + anchor_area - inter_area  # shape (3,)

    # IoU = interseção / união
    ious = inter_area / (union_area + 1e-6)  # shape (3,)
    return ious

def test_yolo_iou(bb):
    """
    Função de teste para verificar o cálculo de IoU com âncoras.
    """
    anchors = anchors_tf[1]

    ious = yolo_iou(bb, anchors)
    best_anchor_idx = np.argmax(ious)
    print("IoUs calculadas:", ious.numpy())
    print("Melhor âncora index:", best_anchor_idx)

    print("Melhor âncora:", anchors[best_anchor_idx][0].numpy(), anchors[best_anchor_idx][1].numpy())

if __name__ == "__main__":
    bboxes = [[0, 0.40865384615384615, 0.41225961538461536, 0.125, 0.5036057692307693],
          [0, 0.10817307692307693, 0.390625, 0.1502403846153846, 0.24158653846153846],
          [0, 0.38461538461538464, 0.4795673076923077, 0.14182692307692307, 0.24158653846153846]
        ]
    print("Bounding Box:", bboxes[1][3:])
    test_yolo_iou(bboxes[1][3:])
