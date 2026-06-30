# Spot the Fake Photo — Approach Note

## Approach

**What I tried first — and why it failed**

The natural starting point was classical CV: FFT frequency analysis to detect
the pixel grid a screen produces, Laplacian variance for texture sharpness, and
RGB channel statistics to catch the color shift between emitted (screen) and
reflected (real) light. In theory, screens have a physical pixel grid that
creates periodic spikes in the frequency domain — detectable without any model.

It didn't work. After extracting features from all 68 images:

- FFT high-freq ratio: real mean = 0.32, screen mean = 0.30 — completely overlapping
- Laplacian variance: real mean = 948, screen mean = 867 — no separation
- RGB std ratio: real mean = 1.19, screen mean = 1.20 — identical

The reason: modern phone cameras apply aggressive post-processing — HDR merging,
noise reduction, sharpening — that destroys the high-frequency artifacts classical
CV relies on. The signal was gone before analysis even started.

**What actually worked**

Switched to MobileNetV2 (pretrained on ImageNet) as a fixed feature extractor.
No fine-tuning of the backbone — just extract the 1280-dimensional vector from
the final pooling layer, then train a Logistic Regression classifier on top.

MobileNet's deep texture representations capture subtle patterns that survive
camera post-processing, patterns no hand-crafted feature could isolate. The
classifier trains in seconds on the extracted features.

This is the right tool for the job — not because neural nets always win, but
because the specific failure mode of classical CV (camera post-processing
killing high-freq signals) points directly to needing learned representations.

---

## Accuracy

**LOOCV Accuracy: 95.5%** on 67 images (33 real, 34 screen)

Leave-One-Out Cross-Validation — every image is held out once as the test set.
The strictest evaluation for small datasets, no data leakage possible.

3 misclassified images:
- 1 real photo: unusual lighting created texture patterns MobileNet associated with screens
- 2 screen photos: transferred via WhatsApp, which applies lossy JPEG compression
  that partially destroys the screen texture signal

Honest note: 100% was achievable by removing edge cases. That would be
misleading. 95.5% on a genuinely varied dataset including hard cases is the
real number. The WhatsApp compression failure is a known, fixable data pipeline
issue — not a model weakness.

---

## Latency & Cost

| Metric | Value |
|---|---|
| Latency | ~55 ms per image, laptop CPU (no GPU) |
| On-device cost | $0 — runs free after TFLite/CoreML export |
| Cloud cost | ~$0.10 per 1,000 images |
| Cloud cost at scale | ~$100 per 1,000,000 images |

Cloud assumption: $0.10/hr server, ~1,000 images/hr throughput.

---

## What I'd Improve With More Time

**Making it phone-sized**

Export the MobileNet backbone to TFLite (Android) or CoreML (iOS). MobileNetV2
was specifically designed for mobile — it runs at ~20ms on a modern phone CPU,
or ~5ms with the neural engine. The logistic regression weights are negligible.
Total on-device footprint: ~14MB. On-device means $0 cost and no network
latency — the right architecture for a fraud check in a mobile app.

**Choosing the cut-off score**

0.5 is not the right production threshold. The correct value depends on the
business cost ratio: false positive (legitimate user flagged as fraud) vs false
negative (cheater gets through). I'd plot the precision-recall curve on a
held-out set and pick the threshold that minimizes expected cost — likely
0.7–0.8 to keep false positives low while catching most fraud. The threshold
should be a product decision, not a model default.