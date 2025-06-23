import numpy as np
import glob
import os

def read_annotations(dataset_path):
    boxes = []
    # Lê anotações de train e valid
    for folder in ["train", "valid"]:
        annotation_files = glob.glob(os.path.join(dataset_path, folder, "*.txt"))
        for file in annotation_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        _, x_center, y_center, width, height = parts[:5]
                        boxes.append([width, height])  # Pega apenas width e height
    return np.array(boxes)

def kmeans_anchors(boxes, k=6, max_iter=100):
    def iou(box, clusters):
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]
        return intersection / (box_area + cluster_area - intersection)

    n = boxes.shape[0]
    distances = np.zeros((n, k))
    last_clusters = np.zeros((n,))

    clusters = boxes[np.random.choice(n, k, replace=False)]  # Inicializa clusters aleatórios

    for _ in range(max_iter):
        for i in range(n):
            distances[i] = 1 - iou(boxes[i], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        last_clusters = nearest_clusters
        for j in range(k):
            clusters[j] = np.median(boxes[nearest_clusters == j], axis=0)
    return clusters

# Caminho para o dataset (ajuste conforme sua estrutura)
dataset_path = "flipping_bird_dataset"
boxes = read_annotations(dataset_path)
anchors = kmeans_anchors(boxes, k=6)

# Ordena as anchors por área (opcional, mas útil para YOLO)
anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

# Converte para pixels (assumindo imagens 416x416)
print("Anchors calculadas (width, height) - Normalizadas:")
print(anchors)
print("\nAnchors em pixels (416x416):")
print(", ".join([f"{int(anchor[0] * 416)}, {int(anchor[1] * 416)}" for anchor in anchors]))