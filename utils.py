import tensorflow as tf
import numpy as np

def conditional_iou(boxes1, boxes2, box_format='corner'):
    """
    Calcula o IoU entre dois conjuntos de caixas delimitadoras com um formato especificado.
    
    Args:
        boxes1: Tensor de shape [N, 4] contendo N caixas delimitadoras.
        boxes2: Tensor de shape [M, 4] contendo M caixas delimitadoras.
        box_format: 'mid' para (x_center, y_center, width, height) ou 'corner' para (x1, y1, x2, y2).
    
    Returns:
        Tensor de shape [N, M] com os valores de IoU entre cada par de caixas.
    """
    # Converter para formato 'corner' (x1, y1, x2, y2) se estiver em 'mid'
    if box_format == 'mid':
        boxes1 = tf.concat([
            boxes1[..., :2] - boxes1[..., 2:] / 2.0,  # x1, y1
            boxes1[..., :2] + boxes1[..., 2:] / 2.0    # x2, y2
        ], axis=-1)
        
        boxes2 = tf.concat([
            boxes2[..., :2] - boxes2[..., 2:] / 2.0,  # x1, y1
            boxes2[..., :2] + boxes2[..., 2:] / 2.0    # x2, y2
        ], axis=-1)
    
    # Broadcasting para comparar todos os pares (N, M)
    boxes1 = tf.expand_dims(boxes1, axis=1)  # [N, 1, 4]
    boxes2 = tf.expand_dims(boxes2, axis=0)  # [1, M, 4]
    
    # Calcular interseção
    x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    
    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    
    # Calcular áreas individuais
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # IoU = Interseção / União
    union = area1 + area2 - intersection
    iou = tf.math.divide_no_nan(intersection, union)  # Evita divisão por zero
    
    return iou

def yolo_iou(bb, anchors):
    """
    Versão mais segura que aceita tanto numpy arrays quanto tensores TF
    """
    try:
        # Converter para numpy primeiro
        bb_np = np.array(bb, dtype=np.float32)
        anchors_np = np.array(anchors, dtype=np.float32)
        
        # Verificar shape
        if bb_np.shape != (5,):
            print(f"BB shape inválido: {bb_np.shape}")
            return np.zeros((3, 3), dtype=np.float32)
            
        # Extrair w e h (índices 3 e 4)
        bb_w, bb_h = bb_np[3], bb_np[4]
        
        # Reformatar âncoras
        flat_anchors = anchors_np.reshape(-1, 2)
        
        # Cálculo do IoU
        bb_area = bb_w * bb_h
        anchors_areas = flat_anchors[:, 0] * flat_anchors[:, 1]
        
        inter_w = np.minimum(bb_w, flat_anchors[:, 0])
        inter_h = np.minimum(bb_h, flat_anchors[:, 1])
        intersection = inter_w * inter_h
        
        union = bb_area + anchors_areas - intersection
        iou = np.divide(intersection, union, where=union!=0)
        
        return iou.reshape(3, 3)
        
    except Exception as e:
        print(f"Erro em yolo_iou: {e}")
        return np.zeros((3, 3), dtype=np.float32)
