import tensorflow as tf
print(f"TF Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
tf.debugging.set_log_device_placement(True)

# Teste operação na GPU
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
    print("\n✅ Multiplicação de matrizes na GPU:")
    print(c.numpy())
except Exception as e:
    print("\n❌ Falha ao usar GPU:", e)