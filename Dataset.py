from config import anchors_tf, NUM_CLASSES, IMAGE_SIZE, DATASET_PATH
from utils import yolo_iou
import os
import numpy as np
import tensorflow as tf
import cv2

class YOLODatasetTF(tf.data.Dataset):
    def __new__(cls, root_dir, subset="train", image_size=416):
        subset_dir = os.path.join(root_dir, subset)
        img_paths = sorted(tf.io.gfile.glob(os.path.join(subset_dir, "*.jpg")))
        label_paths = sorted(tf.io.gfile.glob(os.path.join(subset_dir, "*.txt")))

        def parse_example(img_path, label_path):
            # Debug: mostrar paths
            tf.print("[DEBUG] Processando imagem:", img_path)
            tf.print("[DEBUG] Label correspondente:", label_path)
            
            image = tf.py_function(cls._load_image, [img_path], tf.float32)
            bboxes = tf.py_function(cls._load_bboxes, [label_path], tf.float32)

            image.set_shape([image_size, image_size, 3])
            bboxes.set_shape([None, 5])

            # Debug: mostrar shapes intermediários
            tf.print("[DEBUG] Shape da imagem:", tf.shape(image))
            tf.print("[DEBUG] Número de bboxes:", tf.shape(bboxes)[0])

            target_small = tf.py_function(
                func=cls._generate_target,
                inp=[bboxes, anchors_tf[0], 13, 32],
                Tout=tf.float32
            )
            target_small.set_shape([13, 13, 3, 5 + NUM_CLASSES])
            
            target_medium = tf.py_function(
                func=cls._generate_target,
                inp=[bboxes, anchors_tf[1], 26, 16],
                Tout=tf.float32
            )
            target_medium.set_shape([26, 26, 3, 5 + NUM_CLASSES])

            return image, (target_small, target_medium)

        return tf.data.Dataset.from_tensor_slices((img_paths, label_paths)).map(
            parse_example, num_parallel_calls=tf.data.AUTOTUNE
        )

    @classmethod
    def _load_image(cls, path):
        path_str = path.numpy().decode("utf-8")
        tf.print("[DEBUG] Carregando imagem:", path_str)
        
        image = cv2.imread(path_str)
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {path_str}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0
        
        tf.print("[DEBUG] Imagem carregada - min:", np.min(image), "max:", np.max(image))
        return image.astype(np.float32)

    @classmethod
    def _load_bboxes(cls, label_path):
        path_str = label_path.numpy().decode("utf-8")
        tf.print("[DEBUG] Carregando bboxes de:", path_str)
        
        bboxes = []
        with open(path_str, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([class_id, x_center, y_center, width, height])
                
        tf.print("[DEBUG] Número de bboxes encontrados:", len(bboxes))
        return np.array(bboxes, dtype=np.float32)

    @classmethod
    def _generate_target(cls, bboxes, anchors, grid_size, stride):
        targets = np.zeros((grid_size, grid_size, 3, 5 + NUM_CLASSES), dtype=np.float32)
        tf.print("[DEBUG] Gerando target para grid_size:", grid_size)

        for bb in bboxes:
            class_id, x, y, w, h = bb
            if class_id == 0 and x == 0 and y == 0 and w == 0 and h == 0:
                continue

            ious = yolo_iou(bb, anchors)
            tf.print(f"[DEBUG] IOUs calculadas: {ious}")
            best_anchor_idx = np.argmax(ious)
            tf.print(f"[DEBUG] Melhor anchor index: {best_anchor_idx}")

            cx = x * grid_size
            cy = y * grid_size
            grid_x = int(cx)
            grid_y = int(cy)

            tx = cx - grid_x
            ty = cy - grid_y

            tw = np.log(w / anchors[best_anchor_idx][0] + 1e-6)
            th = np.log(h / anchors[best_anchor_idx][1] + 1e-6)

            targets[grid_y, grid_x, best_anchor_idx, 0] = 1.0
            targets[grid_y, grid_x, best_anchor_idx, 1:5] = [tx, ty, tw, th]
            targets[grid_y, grid_x, best_anchor_idx, 5 + int(class_id)] = 1.0
            print(f"[DEBUG] Target atualizado - grid: ({grid_y},{grid_x}), anchor: {best_anchor_idx}, class: {class_id}")
            tf.print(f"[DEBUG] Bbox processado - grid: ({grid_y},{grid_x}), anchor: {best_anchor_idx}, class: {class_id}")

        return targets

def configure_device():
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️ Nenhuma GPU detectada. Verifique:")
        print("1. Drivers NVIDIA instalados (execute 'nvidia-smi')")
        print("2. CUDA/cuDNN instalados corretamente")
        print("3. Pacote tensorflow-gpu instalado")
        print("➡️ Continuando com CPU...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return 'CPU'
    
    try:
        # Limitar o uso de memória da GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ {len(gpus)} GPU(s) física(s), {len(logical_gpus)} GPU(s) lógica(s)")
        return 'GPU'
    except RuntimeError as e:
        print(f"❌ Erro ao configurar GPU: {e}")
        return 'CPU'

if __name__ == "__main__":
    print("\n🛠️ Configuração do Sistema:")
    device = configure_device()
    
    # Forçar a execução em CPU se necessário
    if device == 'CPU':
        with tf.device('/CPU:0'):
            try:
                train_dataset = YOLODatasetTF(DATASET_PATH, subset="train")
                # ... resto do seu código
            except Exception as e:
                print(f"Erro durante execução: {e}")
    else:
        try:
            train_dataset = YOLODatasetTF(DATASET_PATH, subset="train")
            # ... resto do seu código
        except Exception as e:
            print(f"Erro durante execução: {e}")