import tensorflow as tf
from tensorflow.python.client import device_lib

# Check num GPUs
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(f"\nAvailable GPUs: {gpus}\n")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())