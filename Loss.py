import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

from config import *

def yolo_tiny_loss(y_true, y_pred, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Versão corrigida para formato (batch_size, 26, 26, 18)
    Onde 18 = B*(5+C) com B=3 anchors e C=1 classe
    """
    S = 26  # Grid size
    B = 3   # Número de anchors
    C = 1   # Número de classes
    
    # print("\n=== Formato dos Inputs ===")
    # print(f"y_true shape: {y_true.shape}")
    # print(f"y_pred shape: {y_pred.shape}")
    
    # Redimensiona para (batch_size, S, S, B, 5 + C)
    pred = tf.reshape(y_pred, [-1, S, S, B, 5 + C])
    true = tf.reshape(y_true, [-1, S, S, B, 5 + C])

    # Extrai componentes
    pred_obj = pred[..., 0]      # obj_score (shape [batch_size, S, S, B])
    pred_box = pred[..., 1:5]    # box coords (shape [batch_size, S, S, B, 4])
    pred_class = pred[..., 5:]   # class probs (shape [batch_size, S, S, B, C])

    true_obj = true[..., 0]      # obj_score
    true_box = true[..., 1:5]    # box coords
    true_class = true[..., 5:]   # class probs

    # Máscaras
    obj_mask = true_obj          # shape [batch_size, S, S, B]
    noobj_mask = 1 - obj_mask

    # perda de não-objetos
    no_obj_loss = lambda_noobj * tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(
            true_obj[noobj_mask > 0],
            pred_obj[noobj_mask > 0],
            from_logits=True
        )
    )

    #print(f"Perda de Não-Objetos: {no_obj_loss:.4f}")
    # Perda de objetos
    obj_loss = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(
            true_obj[obj_mask > 0],
            pred_obj[obj_mask > 0],
            from_logits=True
        )
    )

    #print(f"Perda de Objetos: {obj_loss:.4f}")

    # Perda de coordenadas
    box_loss = lambda_coord * tf.reduce_sum(
        tf.square(true_box[obj_mask > 0] - pred_box[obj_mask > 0])
    )
    #print(f"Perda de Coordenadas: {box_loss:.4f}")

    # Perda de classes
    class_loss = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(
            true_class[obj_mask > 0],
            pred_class[obj_mask > 0],
            from_logits=True
        )
    )
    #print(f"Perda de Classes: {class_loss:.4f}")
    
    total_loss = no_obj_loss + obj_loss + box_loss + class_loss
    #print(f"Perda Total: {total_loss:.4f}\n")
    return total_loss