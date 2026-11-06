"""Tasks 2.2 and 2.3: the two models and helpers to freeze and unfreeze layers.

Both models accept raw pixels in 0..255 and scale them inside the model. This means
the saved .keras file carries its own preprocessing, so the Flask app only has to
decode and resize an upload the same way the training pipeline does.
"""
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------- baseline CNN
def build_baseline(img_size=224, dropout=0.4, filters=(32, 64, 128, 256)):
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)  # 0..255 -> 0..1
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


# ------------------------------------------------------------ transfer learning
def build_transfer(img_size=224, dropout=0.3, weights="imagenet"):
    """MobileNetV2 without its classifier, frozen, plus our own small head."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    # MobileNetV2 was trained on pixels scaled to -1..1, which is exactly what
    # keras.applications.mobilenet_v2.preprocess_input does: x / 127.5 - 1.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    base = keras.applications.MobileNetV2(
        include_top=False, weights=weights, input_shape=(img_size, img_size, 3)
    )
    base.trainable = False
    x = base(x, training=False)  # training=False keeps BatchNorm statistics fixed
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def get_base(model):
    """Return the nested pretrained model, or None for the baseline CNN."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


def freeze_base(base):
    base.trainable = False


def unfreeze_last_block(base, first_layer="block_16_expand"):
    """Unfreeze the last inverted residual block of MobileNetV2 (block_16) plus the
    final 1x1 convolution. BatchNorm layers stay frozen on purpose: with a small
    dataset, updating their statistics tends to hurt more than it helps."""
    base.trainable = True
    reached = False
    for layer in base.layers:
        if layer.name == first_layer:
            reached = True
        layer.trainable = reached and not isinstance(layer, layers.BatchNormalization)


def unfreeze_last_n(base, n):
    """Unfreeze the last n layers of the base (used by the tuner). BatchNorm stays frozen."""
    base.trainable = n > 0
    for layer in base.layers:
        layer.trainable = False
    if n > 0:
        for layer in base.layers[-n:]:
            layer.trainable = not isinstance(layer, layers.BatchNormalization)


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ------------------------------------------------------------------- reporting
def layer_table(model) -> pd.DataFrame:
    """Output shape and parameter count per layer, for the 2.2 report table."""
    rows = []
    for layer in model.layers:
        try:
            shape = str(tuple(layer.output.shape))
        except Exception:
            shape = "?"
        rows.append({"layer": layer.name, "type": type(layer).__name__,
                     "output_shape": shape, "params": layer.count_params()})
    return pd.DataFrame(rows)


def trainable_report(model) -> str:
    """Plain text list of layers whose weights will be updated, for the 2.3 report."""
    lines = []
    total = 0
    for layer in model.layers:
        group = layer.layers if isinstance(layer, keras.Model) else [layer]
        for sub in group:
            n = sum(int(w.shape.num_elements()) for w in sub.trainable_weights)
            if n:
                lines.append(f"{sub.name:35s} {type(sub).__name__:22s} {n:>10,d}")
                total += n
    head = f"Trainable layers: {len(lines)}   trainable parameters: {total:,}\n"
    return head + "\n".join(lines)
