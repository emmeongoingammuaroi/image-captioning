import os
import pickle

import numpy as np
import tensorflow as tf

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

attention_features_shape = 64
max_length = 33
units = 512

encoder = tf.keras.models.load_model('./model/attention/encoder')
decoder = tf.keras.models.load_model('./model/attention/decoder')

with open('./model/attention/word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)
with open('./model/attention/index_word.pickle', 'rb') as handle:
    index_word = pickle.load(handle)
with open('./model/attention/img_to_cap.pickle', 'rb') as handle:
    img_to_cap = pickle.load(handle)
with open('./model/attention/img_name_val.pickle', 'rb') as handle:
    img_name_val = pickle.load(handle)
with open('./model/attention/cap_val.pickle', 'rb') as handle:
    cap_val = pickle.load(handle)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    # Delete temp file
    if '/temp/' in image_path:
        os.remove(image_path)
    return img, image_path


def evaluate_attention(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = tf.zeros((1, units))

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(index_word[predicted_id])
        if index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot