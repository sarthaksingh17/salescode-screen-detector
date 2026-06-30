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

With only 67 images and 1280 features per image, a complex model would overfit
badly. Logistic Regression with L2 regularization keeps things simple and
penalizes any single feature from dominating, which matters a lot when you
have way more features than data points.

## Accuracy

95.5% accuracy using Leave-One-Out Cross-Validation on 67 images (33 real, 34
screen). With a dataset this small, a normal train/test split leaves you with
a tiny, unstable test set — LOOCV tests every single image as its own held-out
case and averages the result, which felt like the fairer way to evaluate.

3 images got misclassified — one real photo with unusual lighting that
apparently looked screen-like to the model, and two screen photos that had
been compressed through WhatsApp before I used them, which seems to have
degraded the texture signal enough to throw the model off.

I could've just deleted those 3 and reported 100%, but that felt dishonest.
95.5% on a dataset that includes some messy real-world cases is a more
truthful number, and the WhatsApp compression issue is something I can
actually explain and fix going forward — it's a data pipeline problem, not a
model problem.

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

## What I'd Improve With More Time

For keeping accuracy high as people figure out how to game it, I'd want an
ensemble approach — classical CV signals for the easy, obvious cases (fast and
free), and MobileNet for the harder ones it can't catch. I'd also retrain
periodically on new fraud examples as they get caught, since a static model
will eventually fall behind whatever new tricks people try.

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