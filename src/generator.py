import tensorflow as tf
encoder = tf.keras.models.load_model('Models/vae/encoder')
decoder = tf.keras.models.load_model('Models/vae/decoder')

def gen_similar(arr):
    a = encoder.predict(arr)[2]
    b = a + tf.keras.backend.random_normal(shape=a.shape) * .95
    b_dec = decoder.predict(b)
    return tf.math.round(b_dec)
