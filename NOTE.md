# Spot the Fake Photo — Approach Note

## Approach

I started with classical CV because it felt like the right first move — screens
have a physical pixel grid, so theoretically FFT should pick up periodic spikes
in the frequency domain that real photos don't have. I also tried Laplacian
variance for texture and RGB channel stats, since screens emit light while real
photos just reflect it.

None of it worked. I extracted all three features across my full dataset and
the numbers were basically identical between real and screen photos — FFT
high-freq ratio averaged 0.32 for real vs 0.30 for screen, Laplacian variance
948 vs 867, RGB ratio 1.19 vs 1.20. No separation at all.

Turns out modern phone cameras process images a lot more than I expected —
HDR, noise reduction, sharpening — and that processing wipes out exactly the
high-frequency artifacts FFT needs to detect a screen. So by the time I'm
analyzing the image, the signal classical CV depends on is already gone.

So I switched to MobileNetV2, pretrained on ImageNet, used purely as a feature
extractor — no fine-tuning. I run each image through it and take the
1280-dimensional output from the last pooling layer, then train a Logistic
Regression on top of those features. MobileNet has already learned rich
texture representations from millions of images, so it picks up on subtle
patterns that survive the camera's post-processing — patterns I couldn't
isolate manually with FFT or Laplacian math.

Since 1280 features for just 67 images is a recipe for overfitting, a complex
model would just memorize the noise. Logistic Regression with L2 regularization
keeps things simple and stops any single feature from dominating. I also tossed
in a PCA step to boil those 1280 features down to just the ones that explain 95%
of the variance (usually around 50-60 components). This stops the model from
finding a perfect but meaningless boundary just because it has so many dimensions
to play with.

### Why Classical CV Failed

Looking back at why FFT, Laplacian, and the rest failed completely (near-total
overlap between real and screen images), it makes total sense now. Modern phone
cameras aggressively process images — HDR tone mapping, noise reduction,
computational sharpening. All that processing wipes out the exact high-frequency
screen artifacts (like pixel grids and moiré) that I was trying to detect. By the
time I get the JPEG, the signal is just gone.

I could have tried hunting for chromatic aberration or JPEG block artifacts instead,
but those require strictly controlled lighting and angles that you just won't
get in the real world. The deep feature approach side-steps this entirely by just
learning whatever messy patterns actually survive the camera's processing pipeline.

### Data Augmentation

To help the model generalize better, I threw in some augmentations: flips, a bit
of Gaussian blur, brightness tweaks, and a JPEG compression simulation. I added the
JPEG one specifically because WhatsApp compression caused 2 of my 3 original
misclassifications, so simulating it teaches the model to handle that kind of garbage
data. I dropped geometric stuff like rotation since it didn't really matter for telling
a screen from a real photo.

But I had to be super careful with cross-validation here. I used a grouped split
(`StratifiedGroupKFold`) so that all augmented versions of the same original photo
always stay in the same fold. If I hadn't done that, augmented copies would leak
across the train/test split, basically acting as near-duplicates and artificially
inflating the accuracy to a fake 100%.

## Accuracy

On the raw 67 images (without PCA), LOOCV accuracy was 95.5%. When I added PCA
(53 components explaining 95% of the variance), it actually bumped up to 97.0% —
the PCA step managed to strip out some noisy dimensions that were tripping up a
borderline case.

With the data augmentations added in (giving me over 400 samples) and evaluated
cleanly with the grouped 10-fold CV to prevent leakage, the accuracy sits at a
solid **94.3%**. The mistakes it makes all stem from the same 4 tricky original
images (like the heavily WhatsApp-compressed ones). That actually proves the grouped
CV is doing its job perfectly: when an image is hard, all its augmented variants
end up being hard too.

I also set up a nested cross-validation loop to search for the best regularization
strength (the C parameter) for each fold, just to make sure I wasn't artificially
tuning it on the test data.

### A Note on Training Confidence

One weird thing I noticed: when tested on its own training data, the model is always
100% confident (scoring ~0.05 for real, ~0.96 for screens). Turns out that's totally
normal. With this many features and so few samples, logistic regression will almost
always find a perfect separating hyperplane. It's not necessarily overfitting, it
just means you can't trust the training accuracy — the held-out CV score is the
only one that actually matters.

## Latency & Cost

- Latency: ~55ms per image on my laptop CPU, no GPU involved
- On-device cost: $0, since this would run locally on a phone
- Cloud cost: roughly $0.10 per 1,000 images (assuming a $0.10/hr server
  doing about 1,000 images/hr), so around $100 per million images

## The Demo

I also built a small live demo — a FastAPI backend serving a webpage that
uses the phone/laptop camera, sends frames to `predict.py`, and shows the
fraud score updating in real time. Went with FastAPI over Flask since it's
the more current choice for Python APIs — async support out of the box, and
auto-generated docs if this ever needed to grow past a couple of routes. For
something this small either would've worked fine, but FastAPI felt like the
better habit to build given where this could eventually go.

demo - https://drive.google.com/file/d/1JRw4XHqkljt5Rsj2cNKm6QPRQEFcmP9B/view?usp=sharing

## What I'd Improve With More Time

For making it phone-ready, I'd export the MobileNet backbone to TFLite for
Android or CoreML for iOS. MobileNetV2 is built for exactly this kind of
deployment, so it should run around 20ms on a phone CPU and even faster with
a neural engine. It's already a mobile-friendly architecture, but if I needed
it even smaller or faster, I'd quantize the weights from 32-bit to 8-bit
precision, which cuts the model size roughly 4x with only a small accuracy
trade-off. The logistic regression part barely adds any overhead either way,
so the whole pipeline would stay small, fast, and free to run on-device with
no network call needed.

For the cutoff score, 0.5 is just a default and not something I'd actually
ship with. The right threshold depends on what matters more to the business —
catching every fraud attempt or avoiding false flags on real users. I'd want
to look at a precision-recall curve on a larger held-out set and pick
something like 0.7-0.8 depending on how costly each type of mistake actually
is in practice.

