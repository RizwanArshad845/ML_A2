"""Task 2.1: the tf.data input pipeline.

Stages, in order:
  1. from_tensor_slices   file paths and labels from the split list
  2. map(decode, resize)  read bytes, decode to 3 channels, resize to a square
  3. cache                keep the decoded, resized images in memory after epoch 1
  4. shuffle              training only, new order every epoch
  5. map(augment)         training only, random changes applied fresh every epoch
  6. batch                stack images into tensors
  7. prefetch             prepare the next batch while the model trains on this one

Images leave the pipeline as float32 values in 0..255. Scaling to the range the
network expects happens inside the model, so a saved model accepts raw pixels.
"""
import pandas as pd
import tensorflow as tf

from config import DATA_ROOT
from preprocess import decode_and_resize_bytes

AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(path, img_size: int, resize_mode: str = "pad"):
    """Read one file and return a float32 tensor of shape (img_size, img_size, 3).
    The decode/resize itself lives in preprocess.py, shared with the Flask app."""
    return decode_and_resize_bytes(tf.io.read_file(path), img_size, resize_mode)


def make_augment(level: str, img_size: int):
    """Return a function that randomly changes an image, or None for level 'none'."""
    if level == "none":
        return None

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1 * 255)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        if level == "strong":
            img = tf.image.random_saturation(img, 0.8, 1.2)
            scale = tf.random.uniform([], 0.8, 1.0)
            crop = tf.cast(scale * img_size, tf.int32)
            img = tf.image.random_crop(img, [crop, crop, 3])
            img = tf.image.resize(img, (img_size, img_size))
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    return augment


def build_dataset(csv_path, img_size=224, batch_size=32, resize_mode="pad",
                  augment="basic", training=False, seed=42, root=DATA_ROOT):
    df = pd.read_csv(csv_path)
    paths = [str(root / p) for p in df["path"]]
    labels = df["label"].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (decode_and_resize(p, img_size, resize_mode), y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
        aug_fn = make_augment(augment, img_size)
        if aug_fn is not None:  # augmentation is never applied to validation or test
            ds = ds.map(aug_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, df


def load_image_array(path, img_size=224, resize_mode="pad"):
    """Single image, same preprocessing as training. Returns shape (1, H, W, 3)."""
    return tf.expand_dims(decode_and_resize(tf.constant(str(path)), img_size, resize_mode), 0)
