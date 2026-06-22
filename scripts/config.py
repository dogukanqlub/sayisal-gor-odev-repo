"""Fixed experiment parameters — do not change mid-run."""

# --- Compression / channel noise (orijinal deneyler) ---
JPEG_QUALITIES = [90, 70, 50, 30, 10]
BLUR_KERNELS = [3, 5, 7, 11]
NOISE_SIGMAS = [5, 10, 20, 40]

# --- Post-processing (sosyal medya / düzenleme simülasyonu) ---
RESIZE_SCALES = [0.75, 0.5, 0.25]  # downscale → upscale (orijinal boyuta)
GAMMA_VALUES = [0.6, 0.8, 1.2, 1.4]
CONTRAST_ALPHAS = [0.7, 1.0, 1.3]
SHARPEN_STRENGTHS = [0.5, 1.0, 2.0]
MEDIAN_KERNELS = [3, 5, 7]
SATURATION_FACTORS = [0.5, 1.0, 1.5]

# --- Adversarial (FGSM, piksel uzayı epsilon) ---
FGSM_EPSILONS = [4, 8, 16, 32]

# Tekrarlanabilirlik
RANDOM_SEED = 42
