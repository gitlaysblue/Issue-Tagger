[
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60394",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60394",
    "number": 60394,
    "title": "TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'",
    "user": {
      "login": "student123"
    },
    "labels": [
      {
        "name": "type:bug"
      }
    ],
    "state": "open",
    "created_at": "2023-08-15T09:24:31Z",
    "body": "I'm getting an error when trying to run my TensorFlow model:\n\n```python\nimport tensorflow as tf\n\ndef my_func():\n    x = None\n    return x - 5  # This causes the error\n\ntf.function(my_func)()\n```\n\nThe error I get is:\n```\nTypeError: unsupported operand type(s) for -: 'NoneType' and 'int'\n```\n\nHow do I fix this issue?"
  },
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60393",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60393",
    "number": 60393,
    "title": "Add support for custom activation functions in LSTM layers",
    "user": {
      "login": "mlresearcher"
    },
    "labels": [
      {
        "name": "type:feature"
      }
    ],
    "state": "open",
    "created_at": "2023-08-15T08:19:15Z",
    "body": "## Feature Request\n\nIt would be great if we could add support for custom activation functions in LSTM layers. Currently, we can only use predefined activation functions, but I need to use a custom function for my research project.\n\n### Proposed API\n\n```python\nclass CustomActivation(tf.keras.layers.Layer):\n    def call(self, inputs):\n        return tf.math.tanh(inputs) * tf.math.sigmoid(inputs)\n\nlstm_layer = tf.keras.layers.LSTM(\n    units=64,\n    activation=CustomActivation(),\n    recurrent_activation='sigmoid'\n)\n```\n\nThis would allow researchers to experiment with different activation functions for their RNN models."
  },
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60392",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60392",
    "number": 60392,
    "title": "How to implement multi-head attention in Tensorflow 2.x?",
    "user": {
      "login": "deeplearner42"
    },
    "labels": [
      {
        "name": "type:question"
      }
    ],
    "state": "open",
    "created_at": "2023-08-15T07:45:01Z",
    "body": "I'm trying to implement a transformer model from scratch using TensorFlow 2.x, but I'm having trouble with the multi-head attention mechanism. Is there a simple way to implement this?\n\nI've looked at `tf.keras.layers.MultiHeadAttention` but I'm not sure how to properly set it up. Are there any good examples or documentation for this?\n\nSpecifically, I'm confused about how to handle the masking and the proper input format."
  },
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60391",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60391",
    "number": 60391,
    "title": "Model randomly crashes during training on large dataset",
    "user": {
      "login": "datasciencestudent"
    },
    "labels": [],
    "state": "open",
    "created_at": "2023-08-15T06:30:42Z",
    "body": "## Bug Description\n\nI've been training a model on a large dataset (about 50GB) and it randomly crashes after a few epochs. There's no consistent error message, but it seems to be related to memory.\n\nHere's my model architecture:\n```python\nmodel = tf.keras.Sequential([\n    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),\n    tf.keras.layers.MaxPooling2D((2, 2)),\n    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),\n    tf.keras.layers.MaxPooling2D((2, 2)),\n    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),\n    tf.keras.layers.MaxPooling2D((2, 2)),\n    tf.keras.layers.Flatten(),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dense(10, activation='softmax')\n])\n```\n\nI'm using a batch size of 64 and training for 20 epochs. The crash typically happens around epoch 7-10.\n\nSystem specs:\n- 16GB RAM\n- NVIDIA GeForce RTX 3080 (10GB VRAM)\n- TensorFlow 2.10.0\n\nAny ideas on how to fix this?"
  },
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60390",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60390",
    "number": 60390,
    "title": "Suggestion: Add built-in support for CBAM attention mechanism",
    "user": {
      "login": "airesearcher"
    },
    "labels": [],
    "state": "open",
    "created_at": "2023-08-15T05:12:15Z",
    "body": "I'd like to suggest adding built-in support for the Convolutional Block Attention Module (CBAM) as described in the paper \"CBAM: Convolutional Block Attention Module\" (https://arxiv.org/abs/1807.06521).\n\nThis would be a valuable addition to TensorFlow's attention mechanisms, particularly for computer vision tasks. I've implemented it myself, but having an official implementation would help standardize usage and ensure optimal performance.\n\nHere's a rough sketch of how the API might look:\n\n```python\n# Channel attention module\nclass ChannelAttention(tf.keras.layers.Layer):\n    def __init__(self, ratio=16):\n        super(ChannelAttention, self).__init__()\n        self.ratio = ratio\n        # ... implementation details\n    \n    def call(self, inputs):\n        # ... implementation\n        return channel_attention * inputs\n\n# Spatial attention module\nclass SpatialAttention(tf.keras.layers.Layer):\n    def __init__(self, kernel_size=7):\n        super(SpatialAttention, self).__init__()\n        self.kernel_size = kernel_size\n        # ... implementation details\n    \n    def call(self, inputs):\n        # ... implementation\n        return spatial_attention * inputs\n\n# Combined CBAM\nclass CBAM(tf.keras.layers.Layer):\n    def __init__(self, ratio=16, kernel_size=7):\n        super(CBAM, self).__init__()\n        self.channel_attention = ChannelAttention(ratio)\n        self.spatial_attention = SpatialAttention(kernel_size)\n    \n    def call(self, inputs):\n        output = self.channel_attention(inputs)\n        output = self.spatial_attention(output)\n        return output\n```\n\nI'd be happy to contribute to implementing this if there's interest."
  },
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60389",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60389",
    "number": 60389,
    "title": "Segmentation fault when using tf.data.Dataset with large files",
    "user": {
      "login": "bugfinder"
    },
    "labels": [
      {
        "name": "type:bug"
      }
    ],
    "state": "open",
    "created_at": "2023-08-15T04:05:33Z",
    "body": "## Bug Report\n\nI'm experiencing a segmentation fault when using `tf.data.Dataset` with large TFRecord files (>2GB). The program crashes without any useful error message.\n\nReproduction steps:\n\n1. Create a large TFRecord file:\n```python\nimport tensorflow as tf\nimport numpy as np\n\nwith tf.io.TFRecordWriter('large_file.tfrecord') as writer:\n    for i in range(100000):  # This creates a file >2GB\n        feature = {\n            'feature': tf.train.Feature(\n                float_list=tf.train.FloatList(value=np.random.rand(10000).astype(np.float32)))\n        }\n        example = tf.train.Example(features=tf.train.Features(feature=feature))\n        writer.write(example.SerializeToString())\n```\n\n2. Try to read the file:\n```python\ndataset = tf.data.TFRecordDataset('large_file.tfrecord')\ndataset = dataset.map(lambda x: tf.io.parse_single_example(\n    x, {'feature': tf.io.FixedLenFeature([10000], tf.float32)}))\n\n# This causes a segmentation fault\nfor item in dataset:\n    print(item['feature'].shape)\n    break\n```\n\nSystem information:\n- TensorFlow version: 2.10.0\n- OS: Ubuntu 20.04\n- Python version: 3.8.10\n\nI've tried with smaller files and it works fine, so it seems to be related to the file size."
  },
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60388",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60388",
    "number": 60388,
    "title": "Can TensorFlow models be deployed on ARM microcontrollers?",
    "user": {
      "login": "embeddeddev"
    },
    "labels": [
      {
        "name": "comp:lite"
      }
    ],
    "state": "open",
    "created_at": "2023-08-15T03:22:19Z",
    "body": "I'm working on a project using an STM32 microcontroller (ARM Cortex-M4) and I'd like to deploy a simple neural network for sensor data processing. Is this possible with TensorFlow?\n\nI've heard about TensorFlow Lite for Microcontrollers, but I'm not sure if it's compatible with my specific hardware or how to get started. Could someone point me to some resources or examples for deploying models on ARM Cortex-M microcontrollers?\n\nSpecifically:\n1. What are the minimum hardware requirements?\n2. How do I convert a regular TensorFlow model to run on this hardware?\n3. Are there any limitations I should be aware of?\n\nThanks in advance!"
  },
  {
    "url": "https://api.github.com/repos/tensorflow/tensorflow/issues/60387",
    "html_url": "https://github.com/tensorflow/tensorflow/issues/60387",
    "number": 60387,
    "title": "Feature Request: Add support for nested dictionaries in Model.fit() with custom training loops",
    "user": {
      "login": "nesteddata"
    },
    "labels": [],
    "state": "open",
    "created_at": "2023-08-15T02:11:44Z",
    "body": "## Feature Request\n\nIt would be helpful if TensorFlow's `Model.fit()` method could natively handle nested dictionaries for inputs and outputs. This would be especially useful for complex models with multiple inputs and outputs that have hierarchical relationships.\n\nCurrently, when using a custom training loop, we need to flatten nested dictionaries or use workarounds that make the code less intuitive and harder to maintain.\n\n### Proposed Behavior\n\n```python\n# Input data as a nested dictionary\ninputs = {\n    'image': {\n        'rgb': tf.random.normal((32, 224, 224, 3)),\n        'depth': tf.random.normal((32, 224, 224, 1))\n    },\n    'metadata': tf.random.normal((32, 10))\n}\n\n# Target data as a nested dictionary\ntargets = {\n    'classification': tf.random.uniform((32, 1), maxval=10, dtype=tf.int32),\n    'regression': {\n        'x': tf.random.normal((32, 1)),\n        'y': tf.random.normal((32, 1))\n    }\n}\n\n# Model that accepts and returns nested dictionaries\nmodel = MyComplexModel()\n\n# This should work without flattening the dictionaries\nmodel.fit(inputs, targets, epochs=10, batch_size=32)\n```\n\nThis would make it much