# Task 2 report notes

## 2.1 Input pipeline

The input pipeline is built with `tf.data` in `src/data.py` (function `build_dataset`). It reads the split lists produced by `src/make_splits.py` (`data/splits/train.csv`, `val.csv`, `test.csv`), where each row holds an image path and its label (1 = AI-generated, 0 = real). The same function builds all three datasets; a `training` flag decides whether the training-only stages run.

**Pipeline stages, in order**

1. **Read the split list (`from_tensor_slices`).** The CSV is loaded and turned into a dataset of (file path, label) pairs. At this point only file names are held in memory, not images.
2. **Decode and resize (`map`).** Each file is read as raw bytes and decoded with `tf.io.decode_image(channels=3)`. Forcing three channels converts the greyscale, palette and RGBA images found in the audit into plain RGB, so every image has the same shape. The image is then resized to 224 x 224 (see aspect ratio below) and kept as float32 values in the range 0 to 255. This step runs in parallel (`num_parallel_calls=AUTOTUNE`).
3. **Cache (`cache`).** Decoding large originals (some are close to 3000 px wide) is the slowest part of the pipeline. After the first epoch the decoded and resized images are kept in memory, so later epochs skip the disk and the decoder. The cache is placed before augmentation on purpose: if it came after, the same augmented images would be replayed every epoch.
4. **Shuffle (training only).** The training set is shuffled with a buffer equal to its full size (681 images), using a fixed seed and a new order every epoch. This stops the model from learning anything from the order of the files. Validation and test are not shuffled, so their predictions line up with the rows of their CSV files.
5. **Augment (training only).** Every training image gets a random horizontal flip, a random brightness change of up to ±10% and a random contrast change between 0.9 and 1.1. The changes are drawn fresh every epoch, so the model rarely sees exactly the same picture twice, which reduces overfitting on a small dataset. The changes are mild on purpose: strong blurring or colour shifts could destroy the subtle texture and colour cues that separate AI images from real ones. Validation and test images are never augmented, because they must measure performance on images as they really are.
6. **Batch (`batch(32)`).** Images are grouped into batches of 32, giving tensors of shape (32, 224, 224, 3). Batch size 32 is a common default that fits easily in memory on a CPU.
7. **Prefetch (`prefetch(AUTOTUNE)`).** While the model trains on one batch, the pipeline prepares the next one in the background, so the model does not sit idle waiting for data.

Pixel scaling is not a pipeline stage. Each model scales the raw 0 to 255 pixels itself with a `Rescaling` layer (to 0..1 for the baseline CNN, to -1..1 for MobileNetV2). Because the scaling is stored inside the saved model, the Flask app only has to decode and resize an upload the same way, and training and serving preprocessing cannot drift apart.

**Image size.** I chose 224 x 224 because it is the input size MobileNetV2 was pretrained on, so the pretrained filters see objects at the scale they learned. Using the same size for the baseline CNN keeps the comparison between the two models fair. It is also small enough to train on a CPU in reasonable time.

**Aspect ratio.** The audit showed that images come in many shapes (for example 880 x 440 landscapes next to tall portraits). Squeezing them all into a square would stretch or squash their content, which could create or hide the very artefacts the model is meant to detect. I therefore used `tf.image.resize_with_pad`: the image is scaled so its longer side becomes 224 px, keeping its proportions, and the remaining space is filled with black bars. The cost is that wide or tall images use fewer real pixels and carry black borders. A plain stretch (`resize_mode="stretch"`) is still available as an option for the experiment comparison in Task 3.

**Figure.** `reports/figures/augmented_batch.png` shows the first 16 images of one augmented training batch (made with `src/preview_batch.py`). The black padding bars and the varied orientations and brightness are visible.

**What went wrong / limitation.** The brightness augmentation is applied after padding, so it also lightens the black bars slightly (they appear dark grey in some images in the figure). This is harmless but means the padding is not perfectly constant. Doing augmentation before padding would avoid it.
