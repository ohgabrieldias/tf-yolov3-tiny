import tensorflow as tf
from config import NUM_CLASSES, LAMBDA_COORD, LAMBDA_NOOBJ

@tf.function
def yolo_tiny_loss(y_pred, y_true, S, anchors):

    B = len(anchors)   # Número de anchors

    pred = tf.reshape(y_pred, [-1, S, S, B, 5 + NUM_CLASSES])
    true = tf.reshape(y_true, [-1, S, S, B, 5 + NUM_CLASSES])


    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    mse = tf.keras.losses.MeanSquaredError(reduction='none')
    sigmoid = tf.keras.activations.sigmoid

    # Máscaras de objeto e não objeto
    obj = tf.equal(true[..., 0], 1.0)
    noobj = tf.equal(true[..., 0], 0.0)

    # Reshape dos anchors para broadcasting
    anchors = tf.cast(tf.reshape(anchors, [1, 1, 1, 3, 2]), tf.float32)  # (1, num_anchors, 1, 1, 2) #pw * exp(tw)

    # ======================= #
    #   FOR NO OBJECT LOSS    #
    # ======================= #
    noobj_loss_vals = bce(true[..., 0:1], pred[..., 0:1])
    no_obj_loss = tf.reduce_mean(tf.boolean_mask(noobj_loss_vals, noobj))

    # ==================== #
    #   FOR OBJECT LOSS    #
    # ==================== #




    # x = sigmoid(tx) + cx
    # y = sigmoid(ty) + cy
    # w = pw * exp(tw)
    # h = ph * exp(th)

    box_preds_xy = sigmoid(pred[..., 1:3])                      # normaliza x e y entre 0 e 1
    box_preds_wh = tf.exp(pred[..., 3:5]) * anchors             # calcular w e h com os anchors
    box_preds = tf.concat([box_preds_xy, box_preds_wh], axis=-1) # (x, y, w, h)

    ## 3. Perda para células COM objeto
    # Calcular IoU
    ious = intersection_over_union(
        tf.boolean_mask(box_preds, obj),
        tf.boolean_mask(true[..., 1:5], obj)
    )

    # Objectness loss ponderada pelo IoU
    obj_loss = tf.reduce_sum(
        tf.keras.losses.mse(tf.sigmoid(pred[..., 0:1][obj]), ious * true[..., 0:1][obj])
        
    )

    # ======================== #
    #   FOR BOX COORDINATES    #
    # ======================== #
    # Calcular a perda de coordenadas da caixa delimitadora
    # Aplicar sigmoid nas coordenadas x, y (t_x, t_y)
    tx_ty_pred = tf.sigmoid(pred[..., 1:3])
    tw_th_pred = pred[..., 3:5]

    # Reconstruir a predição com t_x, t_y modificados
    pred_box = tf.concat([tx_ty_pred, tw_th_pred], axis=-1)

    # Transformar o alvo: log da razão w/h com os anchors
    tx_ty_true = true[..., 1:3]
    tw_th_true = tf.math.log(1e-16 + true[..., 3:5] / anchors)
    target_box = tf.concat([tx_ty_true, tw_th_true], axis=-1)

    # Calcular a perda apenas nas posições com objeto
    box_loss = tf.reduce_sum(
        tf.keras.losses.mse(
            tf.boolean_mask(pred_box, obj),
            tf.boolean_mask(target_box, obj)
        )
    )




    total_loss = (
        LAMBDA_NOOBJ * no_obj_loss +
        obj_loss +
        LAMBDA_COORD * box_loss
    )

    tf.print("Losses | no_obj:", no_obj_loss,
                "| obj:", obj_loss, 
                "| box:", box_loss, 
                "| total:", total_loss)
    
    

    return total_loss


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union (IoU) between predicted boxes and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """
    #tf.print("Predicted boxes:", boxes_preds)
    #tf.print("Correct labels:", boxes_labels)
    if box_format == "midpoint":
        # Convert midpoint (x,y,w,h) to corners (x1,y1,x2,y2)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    elif box_format == "corners":
        # Already in corner format (x1,y1,x2,y2)
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    # Calculate intersection coordinates
    x1 = tf.maximum(box1_x1, box2_x1)
    y1 = tf.maximum(box1_y1, box2_y1)
    x2 = tf.minimum(box1_x2, box2_x2)
    y2 = tf.minimum(box1_y2, box2_y2)
    
    # Calculate intersection area (using tf.maximum instead of clamp(0))
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    
    # Calculate box areas
    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    # Calculate union and IoU
    union = box1_area + box2_area - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())  # Using epsilon instead of 1e-6
    #tf.print("IOUS:", iou)
    return iou + 1e-6  # Adding a small value to avoid division by zero

@tf.function
def yolo_loss(y_pred, y_true, S, anchors,
            lambda_noobj=0.5, lambda_obj=1.0, lambda_box=5.0):
    
    B = 3   # Número de anchors

    predictions = tf.reshape(y_pred, [-1, S, S, B, 5 + NUM_CLASSES])
    target = tf.reshape(y_true, [-1, S, S, B, 5 + NUM_CLASSES])

    # Máscaras de objeto e não objeto
    obj = tf.equal(target[..., 0], 1.0)
    noobj = tf.equal(target[..., 0], 0.0)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    mse = tf.keras.losses.MeanSquaredError(reduction='none')
    entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    sigmoid = tf.keras.activations.sigmoid

    # ======================= #
    #   FOR NO OBJECT LOSS    #
    # ======================= #
    noobj_loss_vals = bce(target[..., 0:1], predictions[..., 0:1])
    no_object_loss = tf.reduce_mean(tf.boolean_mask(noobj_loss_vals, noobj))

    # ==================== #
    #   FOR OBJECT LOSS    #
    # ==================== #
    anchors = tf.reshape(anchors, (1, 1, 1, 3, 2))
    box_preds_xy = sigmoid(predictions[..., 1:3])                           # normaliza x e y entre 0 e 1
    box_preds_wh = tf.exp(predictions[..., 3:5]) * anchors                  #pw etw ph eth
    box_preds = tf.concat([box_preds_xy, box_preds_wh], axis=-1)            # (x, y, w, h)

    true_boxes = target[..., 1:5]

    ious = intersection_over_union_tf(tf.boolean_mask(box_preds, obj), tf.boolean_mask(true_boxes, obj))
    ious = tf.stop_gradient(ious)                                           # Não propaga gradiente para o IoU

    obj_sigmoid = sigmoid(predictions[..., 0:1])                            # Sigmoid para a probabilidade de objeto
    obj_loss_vals = mse(obj_sigmoid, ious * target[..., 0:1])               # Calcula a perda de objeto ponderada pelo IoU
    object_loss = tf.reduce_mean(tf.boolean_mask(obj_loss_vals, obj))       # Calcula a perda apenas nas posições com objeto

    # ======================== #
    #   FOR BOX COORDINATES    #
    # ======================== #
    predictions_xy = sigmoid(predictions[..., 1:3])                     # normaliza x e y entre 0 e 1
    target_wh = tf.math.log(1e-16 + (target[..., 3:5] / anchors))       # log da razão w/h com os anchors
    target_adj = tf.concat([predictions_xy, target_wh], axis=-1)        # (x, y, log(w), log(h))

    box_loss_vals = mse(predictions[..., 1:5], target_adj)              # Calcula a perda de coordenadas da caixa delimitadora
    box_loss = tf.reduce_mean(tf.boolean_mask(box_loss_vals, obj))      # Calcula a perda apenas nas posições com objeto


    # ======================== #
    #   FOR CLASS LOSS        #
    # ======================== #
    # pred_classes = tf.boolean_mask(predictions[..., 5:], obj)  # shape: [N_obj, NUM_CLASSES]
    # true_classes = tf.boolean_mask(target[..., 5], obj)        # shape: [N_obj]
    
    # # Convert true_classes to int if needed
    # true_classes = tf.cast(true_classes, tf.int32)
    
    # # Calculate class loss
    # class_loss_vals = entropy(true_classes, pred_classes)
    # class_loss = tf.reduce_mean(class_loss_vals)
    # ======================= #
    #   TOTAL LOSS            #
    # ======================= #
    total_loss = (
        lambda_box * box_loss +
        lambda_obj * object_loss +
        lambda_noobj * no_object_loss
        # class_loss
    )
    #mostra valores com 2 casas decimais
    # no_object_loss = tf.round(no_object_loss * 100) / 100
    # object_loss = tf.round(object_loss * 100) / 100
    # box_loss = tf.round(box_loss * 100) / 100
    # total_loss = tf.round(total_loss * 100) / 100

    tf.print("Losses | no_obj:", no_object_loss,
                "| obj:", object_loss,
                "| box:", box_loss,
                #"| class:", class_loss,
                "| total:", total_loss)

    return total_loss
def intersection_over_union_tf(boxes_preds, boxes_labels):
    # (x, y, w, h) to (x1, y1, x2, y2)
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    x1 = tf.maximum(box1_x1, box2_x1)
    y1 = tf.maximum(box1_y1, box2_y1)
    x2 = tf.minimum(box1_x2, box2_x2)
    y2 = tf.minimum(box1_y2, box2_y2)

    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = box1_area + box2_area - intersection + 1e-6

    return intersection / union
