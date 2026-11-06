"""Image preprocessing shared by training (src/data.py) and serving (app/inference.py).

Training and serving must turn an image into numbers in exactly the same way,
otherwise the model sees inputs at serving time that look nothing like what it
learned from. Keeping the one function here means both sides call the same code.
Pixel scaling (0..255 -> model range) is not done here: it lives inside the model.
"""
import tensorflow as tf


def decode_and_resize_bytes(raw, img_size: int = 224, resize_mode: str = "pad"):
    """Decode encoded image bytes and return float32 (img_size, img_size, 3) in 0..255."""
    # channels=3 turns greyscale, palette and RGBA files into plain RGB.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    if resize_mode == "pad":
        # Keeps the aspect ratio: scale to fit, fill the rest with black bars.
        img = tf.image.resize_with_pad(img, img_size, img_size, antialias=True)
    else:
        # "stretch" squeezes or stretches the image to a square.
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img.set_shape((img_size, img_size, 3))
    return img
