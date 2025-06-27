import tensorflow as tf
import cv2
import numpy as np
from config import *

def preprocess_image(image_path, img_size=416):
    img = cv2.imread(image_path)  # Reads as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    return img_input, original_shape, img

def decode_predictions(pred, anchors, S, conf_thresh=0.6):
    pred = tf.convert_to_tensor(pred)
    pred = tf.reshape(pred, (S, S, 3, -1))
    pred = tf.transpose(pred, (2, 0, 1, 3))  # [3, S, S, 5+num_classes]

    # Reshape anchors to [3, 1, 1, 2] for proper broadcasting
    anchors = tf.reshape(anchors, (3, 1, 1, 2))

    box_xy = tf.sigmoid(pred[..., 1:3])  # tx, ty
    box_wh = tf.exp(pred[..., 3:5]) * anchors  # tw, th
    objectness = tf.sigmoid(pred[..., 0])
    scores = objectness  # se for uma classe só

    grid = tf.meshgrid(tf.range(S), tf.range(S))
    grid = tf.stack(grid, axis=-1)  # [S, S, 2]
    grid = tf.expand_dims(grid, axis=0)  # [1, S, S, 2]
    grid = tf.cast(grid, tf.float32)

    box_xy = (box_xy + grid) / S
    box_wh = box_wh / S  # normalizado
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    boxes = tf.concat([box_x1y1, box_x2y2], axis=-1)

    mask = scores > conf_thresh
    boxes = tf.boolean_mask(boxes, mask)
    scores = tf.boolean_mask(scores, mask)

    return boxes.numpy(), scores.numpy()

def non_max_suppression(boxes, scores, iou_threshold, score_threshold):
    # Filter out boxes with low scores
    keep = scores > score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    
    # Convert to x1,y1,x2,y2 format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with IoU <= threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep], scores[keep]

def draw_boxes(img, boxes, scores):
    h, w = img.shape[:2]
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img

def show_image(img, window_name="Image", wait_time=0):
    """Função segura para mostrar imagens"""
    cv2.imshow(window_name, img)
    key = cv2.waitKey(wait_time)
    if wait_time == 0:  # Se for espera indefinida
        while True:
            key = cv2.waitKey(1)
            if key != -1:  # Se qualquer tecla for pressionada
                break
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Limpeza extra
    return key

MODEL_NAME = "saved_models/yv3tiny_e110_tl0.2_vl1.9.keras"
model = tf.keras.models.load_model(MODEL_NAME, compile=False)

# Print the model summary to verify its structure
model.summary()

img_name = "000010"
img_path = f"{DATASET_PATH}/test/{img_name}.jpg"
img_input, original_shape, original_img = preprocess_image(img_path)

# 3. Make inference
pred_small, pred_medium = model.predict(img_input)

# 4. Scaled anchors
anchors_13 = scaled_anchors_tf[0].numpy()
anchors_26 = scaled_anchors_tf[1].numpy()

# 5. Decode predictions
boxes13, scores13 = decode_predictions(pred_small[0], anchors_13, S=13)
boxes26, scores26 = decode_predictions(pred_medium[0], anchors_26, S=26)

# 6. Combine boxes
all_boxes = np.concatenate([boxes13, boxes26], axis=0)
all_scores = np.concatenate([scores13, scores26], axis=0)

# 6.5 Apply NMS
nms_boxes, nms_scores = non_max_suppression(all_boxes, all_scores, 
                                           iou_threshold=0.1, 
                                           score_threshold=0.6)

# 7. Draw and show
final_img = draw_boxes(original_img.copy(), nms_boxes, nms_scores)
final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
show_image(final_img, "Detections")