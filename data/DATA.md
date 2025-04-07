# Dataset Preparation

- **Wake vision dataset:**  
  The dataset is a large-scale benchmark for TinyML person detection,
  featuring millions of images with both binary labels and fine-grained attribute annotations.  
[WV Website](https://blog.tensorflow.org/2024/12/introducing-wake-vision-new-dataset-for-person-detection-in-tinyml.html)

- **Data Loading:**
  import tensorflow_datasets as tfds
  ds_train, ds_test = tfds.load('wake_vision', split=['train', 'test'], as_supervised=True)

- **Image Augmentations:**
  Apply techniques such as random cropping, flipping, and brightness adjustments using TensorFlow’s tf.image API.
