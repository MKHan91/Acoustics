from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
import os

def deprocess_image(x):
    if np.ndim(x) > 3:
        x = np.squeeze(x)

    x -= x.mean()
    x /=(x.std() + 1e-5)
    x *= 0.1
    # x *= 1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def compile_saliency_function(net, activation_layer):
    input_img = net.input
    layer_dict = dict([(layer.name, layer) for layer in net.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]

    return K.function([input_img, K.learning_phase()], [saliency])

def register_gradient():
    if 'GuidedBackProp' not in ops._gradient_registry._registry:
        @ops.RegisterGradient('GuidedBackProp')
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype=dtype) * tf.cast(op.inputs[0] > 0., dtype=dtype)

def modify_backprop(model, name, ckpt_path, main_name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]

        for num, layer in enumerate(layer_dict):
            if layer.activation == activations.relu:
                layer.activation = tf.nn.relu
            # print('processing: {}/{}'.format(num + 1, len(layer_dict)))
        new_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
        # loaded_model = os.listdir(os.path.join(ckpt_path, '201906_Model', main_name + '_model'))[0]
        # network = load_model(filepath=os.path.join(ckpt_path, '201906_Model', main_name + '_model', loaded_model))
        # activation_layer = network.layers[-9].name
        activation_layer = new_model.layers[-3].name

    return new_model, activation_layer