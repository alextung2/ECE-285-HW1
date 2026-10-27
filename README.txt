ECE 285 Homework 1: convolutional variational autoencoder

Contents
  experiment.py       Dataset preparation, VAE training, generation, and Ignite evaluation
  check_model.py      Model and loss correctness checks
  requirements.txt   Package versions used
  ../results/         Trained checkpoint, raw curves, metrics, generated images
  ../report/          Editable NeurIPS LaTeX source and figures
  build_figures.py    Regenerate the report curves and numeric macros from results

Reproduce (Python 3.12)
  python -m pip install -r requirements.txt
  python check_model.py
Download the original ZIP from:
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Or public API:
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia
Then, from this directory:
  python experiment.py prepare --archive chest-xray.zip --cache cache
  python experiment.py train --cache cache --out ../results --epochs 30 --threads 4
  python experiment.py evaluate --cache cache --out ../results --threads 4
  python build_figures.py

The first evaluation downloads official ImageNet Inception-v3 weights.
The code defaults to CPU. A compatible PyTorch GPU installation can use --device cuda.
Exact numerical agreement is only expected on matching software/hardware.
Preparing the data requires approximately 2.5 GB for the ZIP. The resized cache is small.
The dataset itself is not included in this submission package.

Protocol
Only the top-level chest_xray/train, val, and test archive paths are used because
the archive also includes redundant directory copies. All images are grayscale,
bilinearly resized directly to 64x64, and divided by 255. No augmentation or labels
are used. SHA-256 checks reject exact cross-split duplicates; they cannot establish
patient-level independence. A full file manifest is included in results.
The checkpoint with the lowest eight-draw validation negative ELBO is retained.
No test images are used in optimization or checkpoint selection.
Scores use 624 prior samples against the 624 official test images.
Measured results: FID 206.340037; global IS 1.774624; split IS 1.746289 +/- 0.126933.
The selected checkpoint is epoch 24 of 30. Final training loss is 2393.094518 nats/image.
The evaluation checks Ignite FID against an independent low-rank SVD formula and
checks Ignite IS against the direct KL formula before writing metrics.json.
The saved generated.png displays the first 32 samples without cherry-picking.
FID uses 2048-dimensional average-pool features, while IS uses 1000-class softmax
probabilities, both from torchvision ImageNet Inception-v3, via Ignite metrics.
These FID values are protocol-specific, not interchangeable with TensorFlow/clean-FID.
IS split standard deviation describes variation among ten subsets, not a confidence interval.

Report
The source uses the official neurips_2023.sty with the preprint option; the assignment
does not specify a template year. Run pdflatex main.tex twice from ../report.
The PDF is two pages, including figures and references. Add your name if required
by your course before submitting. No Gradescope submission is performed.
