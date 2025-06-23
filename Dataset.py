from config import *
from utils import yolo_iou
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class YOLODatasetTF(tf.data.Dataset):
    def __new__(cls, root_dir, subset="train", image_size=416):
        subset_dir = os.path.join(root_dir, subset)
        img_paths = sorted(tf.io.gfile.glob(os.path.join(subset_dir, "*.jpg")))
        label_paths = sorted(tf.io.gfile.glob(os.path.join(subset_dir, "*.txt")))

        def parse_example(img_path, label_path):
            # Debug: mostrar paths
            #tf.print("[DEBUG] Processando imagem:", img_path)
            #tf.print("[DEBUG] Label correspondente:", label_path)
            
            image = tf.py_function(cls._load_image, [img_path], tf.float32)
            bboxes = tf.py_function(cls._load_bboxes, [label_path], tf.float32)

            image.set_shape([image_size, image_size, 3])
            bboxes.set_shape([None, 5])

            # Debug: mostrar shapes intermediários
            ##tf.print("[DEBUG] Shape da imagem:", tf.shape(image))
            ##tf.print("[DEBUG] Número de bboxes:", tf.shape(bboxes)[0])

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
        #tf.print("[DEBUG] Carregando imagem:", path_str)
        
        image = cv2.imread(path_str)
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {path_str}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0
        
        ##tf.print("[DEBUG] Imagem carregada - min:", np.min(image), "max:", np.max(image))
        return image.astype(np.float32)

    @classmethod
    def _load_bboxes(cls, label_path):
        path_str = label_path.numpy().decode("utf-8")
        #tf.print("[DEBUG] Carregando bboxes de:", path_str)
        
        bboxes = []
        with open(path_str, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([class_id, x_center, y_center, width, height])
                
        #tf.print("[DEBUG] Número de bboxes encontrados:", len(bboxes))
        return np.array(bboxes, dtype=np.float32)

    @classmethod
    def _generate_target(cls, bboxes, anchors, grid_size, stride):
        targets = np.zeros((grid_size, grid_size, 3, 5 + NUM_CLASSES), dtype=np.float32)
        ##tf.print("[DEBUG] Gerando target para grid_size:", grid_size)
        grid_size = tf.cast(grid_size, tf.float32)
        stride = tf.cast(stride, tf.float32)

        for bb in bboxes:
            class_id, x, y, w, h = bb
            if class_id == 0 and x == 0 and y == 0 and w == 0 and h == 0:
                continue

            ious = yolo_iou(bb[3:], anchors)
            ##tf.print(f"[DEBUG] IOUs calculadas: {ious}")
            best_anchor_idx = np.argmax(ious)
            ##tf.print(f"[DEBUG] Melhor anchor index: {best_anchor_idx}")

            cx = x * IMAGE_SIZE
            cy = y * IMAGE_SIZE

            grid_x = int(cx // stride)  # índice da célula na largura
            grid_y = int(cy // stride) 

            tx = (cx / stride) - grid_x
            ty = (cy / stride) - grid_y

            tw = np.log(w / anchors[best_anchor_idx][0] + 1e-6)
            th = np.log(h / anchors[best_anchor_idx][1] + 1e-6)

            targets[grid_y, grid_x, best_anchor_idx, 0] = 1.0
            targets[grid_y, grid_x, best_anchor_idx, 1:5] = [tx, ty, tw, th]
            targets[grid_y, grid_x, best_anchor_idx, 5 + int(class_id)] = 1.0
            t = tf.convert_to_tensor(targets[grid_y, grid_x, best_anchor_idx])
            t_rounded = tf.round(t * 1e4) / 1e4  # arredonda para 4 casas decimais

            #tf.print("[DEBUG] S =", grid_size,
                    # "| cell =", (grid_y, grid_x),
                    # "| anchor_idx =", best_anchor_idx,
                    # "| target =", t_rounded)


        return targets

def plot_image_with_grid_cells(image, target, grid_size, anchors, stride):
    """
    image: tensor ou np.array (H, W, 3) com valores em [0,1]
    target: np.array (grid_size, grid_size, 3, 5+num_classes)
        camada 0 do canal de objeto (objectness) indica presença de objeto
    anchors: lista de âncoras no formato [(w1, h1), (w2, h2), ...]
    stride: tamanho do passo da grade em pixels (normalmente IMAGE_SIZE/grid_size)

    Exibe a imagem com grade, marca células ocupadas e destaca células dentro dos bboxes previstos.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(image)

    # Tamanho da imagem
    height, width = image.shape[0], image.shape[1]
    
    # Tamanho da célula em pixels
    cell_h = height / grid_size
    cell_w = width / grid_size

    # Plotar linhas da grade
    for i in range(grid_size + 1):
        ax.axhline(i * cell_h, color='gray', linewidth=0.5)
        ax.axvline(i * cell_w, color='gray', linewidth=0.5)

    # Percorrer células para marcar objetos
    for y in range(grid_size):
        for x in range(grid_size):
            for anchor_idx in range(target.shape[2]):
                objectness = target[y, x, anchor_idx, 0]
                if objectness >= 0.5:  # tem objeto nessa célula+âncora
                    # Obter offsets e dimensões previstas
                    tx, ty, tw, th = target[y, x, anchor_idx, 1:5]
                    
                    # Converter para coordenadas absolutas na imagem (usando sua codificação)
                    # 1. Calcular centro do bbox
                    bx = (x + tx) * stride
                    by = (y + ty) * stride
                    
                    # 2. Calcular largura e altura (revertendo o log e multiplicando pela âncora)
                    bw = np.exp(tw) * anchors[anchor_idx][0] * IMAGE_SIZE
                    bh = np.exp(th) * anchors[anchor_idx][1] * IMAGE_SIZE
                   
                    x_min = int(bx - bw / 2)
                    y_min = int(by - bh / 2)

                    print(f"Anchor {anchor_idx} - Cell ({y}, {x}) - BBox: ({bx}, {by}), Size: ({bw}, {bh})")
                    # Desenhar o bbox previsto
                    bbox_rect = plt.Rectangle((x_min, y_min), bw, bh,
                          linewidth=3, edgecolor='red', facecolor='none')
                    ax.add_patch(bbox_rect)
                    
                    # Marcar células que estão dentro deste bbox
                    for cy in range(grid_size):
                        for cx in range(grid_size):
                            # Coordenadas do centro da célula
                            cell_center_x = (cx + 0.5) * cell_w
                            cell_center_y = (cy + 0.5) * cell_h
                            
                            # Verificar se o centro da célula está dentro do bbox
                            if (bx - bw/2 <= cell_center_x <= bx + bw/2 and
                                by - bh/2 <= cell_center_y <= by + bh/2):
                                # Célula dentro do bbox - retângulo amarelo
                                rect = plt.Rectangle((cx * cell_w, cy * cell_h), cell_w, cell_h,
                                                   linewidth=1, edgecolor='yellow', 
                                                   facecolor='yellow', alpha=0.3)
                                ax.add_patch(rect)
                else:
                    # Células sem objeto: retângulo com borda azul fina
                    rect = plt.Rectangle((x * cell_w, y * cell_h), cell_w, cell_h,
                                       linewidth=0.5, edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

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
    
    # Habilitar execução eager para ver os prints
    tf.config.run_functions_eagerly(False)
    
    try:
        dataset = YOLODatasetTF(DATASET_PATH, subset="train") \
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).shuffle(100)


        # Pega apenas o primeiro batch
        for image_batch, (target_small_batch, target_medium_batch) in dataset.take(1):
            # Converte os tensores para numpy
            img_np           = image_batch.numpy()          # shape: (4, H, W, 3)
            target_small_np  = target_small_batch.numpy()   # shape: (4, 13, 13, 3, 5+num_classes)
            target_medium_np = target_medium_batch.numpy()  # se quiser usar a outra escala

            # Loop por cada imagem do batch
            for i in range(img_np.shape[0]):
                image_single  = img_np[i]          # (H, W, 3)

                plot_image_with_grid_cells(
                    image=image_single,
                    target=target_small_np[i],
                    grid_size=13,
                    anchors=anchors_tf[0],  # lista de 3 tuplas (w, h)
                    stride=32
                )

                plot_image_with_grid_cells(
                    image=image_single,
                    target=target_medium_np[i],
                    grid_size=26,
                    anchors=anchors_tf[1],  # lista de 3 tuplas (w, h)
                    stride=16
                )
    except Exception as e:
        print(f"Erro durante execução: {e}")