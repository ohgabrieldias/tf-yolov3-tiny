import tensorflow as tf
print(f"TF Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
tf.debugging.set_log_device_placement(True)
