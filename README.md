# Image Processing with Histogram Techniques

## Overview

This project implements various image processing techniques focusing on grayscale and color image histogram operations. Built using Python, the project leverages popular libraries like OpenCV, Matplotlib, NumPy, and Scikit-image to compute histograms, perform histogram equalization, and apply histogram matching for both grayscale and colored images.

## Features

- **Histogram Computation:** Calculate a 64-bin grayscale histogram for an image.
- **Histogram Equalization:** Enhance contrast in grayscale images by redistributing pixel intensities.
- **Histogram Comparison:** Measure similarity between histograms of two images using the Bhattacharyya Coefficient.
- **Histogram Matching:** Match histograms between grayscale and colored images.

## Prerequisites

Ensure you have Python 3.x installed along with the following libraries:

- `opencv-python`
- `scikit-image`
- `numpy`
- `matplotlib`

Install these dependencies using:

```bash
pip install opencv-python scikit-image numpy matplotlib
```

## Functions

### `part1_a()`

- **Description:** Applies a Laplacian filter to an image to highlight edges.
- **Input:** `moon.png` (grayscale image)
- **Output:** Displays the original and filtered images side by side.

### `part1_b()`

- **Description:** Applies a custom 5x3 filter to an image, averaging specific pixel regions.
- **Input:** `moon.png` (grayscale image)
- **Output:** Displays the original and filtered images side by side.

### `part1_c()`

- **Description:** Applies a custom 3x3 filter to an image, highlighting specific pixel values.
- **Input:** `moon.png` (grayscale image)
- **Output:** Displays the original and filtered images side by side.

### `part1()`

- **Description:** Runs all three image filtering functions from `part1_a`, `part1_b`, and `part1_c`.

### `part2()`

- **Description:** Compares different image filtering techniques: median and Gaussian blurring.
- **Input:** `noisy.jpg` (grayscale image)
- **Output:** Displays the original, median-filtered, and Gaussian-filtered images side by side.

### `part3()`

- **Description:** Restores a damaged image using Gaussian blurring and a damage mask.
- **Input:** `damage_cameraman.png` (damaged image), `damage_mask.png` (mask image)
- **Output:** Displays the original damaged image and the restored image side by side.

### `part4()`

- **Description:** Computes and displays gradients of an image using convolution.
- **Input:** `ex2.jpg` (grayscale image)
- **Output:** Displays the original image, vertical gradient, horizontal gradient, and edge strength (gradient magnitude) images.

### `part5()`

- **Description:** Performs edge detection on an image using the Canny edge detector and compares it to a target image.
- **Input:** `ex2.jpg` (grayscale image), `canny_target.jpg` (target image)
- **Output:** Displays the original image, Gaussian-filtered image, target image, and the edge-detected image.
