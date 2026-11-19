# Waste Classification

A machine learning model that detects the type of a piece of waste, so it can be sorted into the right category: **biodegradable**, **recyclable** or **non-recyclable**.

> Replace the three categories above with the exact classes listed in `labels.txt` if they differ.

This repository contains the full pipeline: training script, saved model, a lightweight TensorFlow Lite version for fast inference, and scripts to run predictions. Automated physical sorting (sensors, a roller and bins) is planned as future work and is **not** part of this version.

## Why this project

Mixed waste is hard to recycle and ends up in landfills. A reliable way to identify waste type is the first step toward automated segregation at homes, schools and community waste centres. This project focuses on that first step: accurate detection of the waste type.

## What it does

1. Loads an image of a waste item. `[FILL: or a live camera frame, whichever detect_waste.py uses]`
2. Runs it through the trained model.
3. Prints the predicted category with its confidence score.

## Classes

The model predicts the classes listed in `labels.txt`:

| Class | Examples |
|-------|----------|
| `[FILL: class 1]` | `[FILL]` |
| `[FILL: class 2]` | `[FILL]` |
| `[FILL: class 3]` | `[FILL]` |

## Project structure

| File / folder | Purpose |
|---------------|---------|
| `data/` | Dataset used for training |
| `train_model.py` | Trains the model and saves it as `waste_model.keras` |
| `waste_model.keras` | Trained Keras model |
| `tflite_model.tflite` | TensorFlow Lite version of the model for lightweight inference |
| `tflite_infer.py` | Runs predictions with the TensorFlow Lite model |
| `detect_waste.py` | `[FILL: one line, e.g. runs detection on an image or webcam feed]` |
| `labels.txt` | Class names, one per line |

## Tech stack

- Python
- TensorFlow / Keras and TensorFlow Lite `[confirm against the imports in your scripts]`
- `[FILL: any other libraries, e.g. NumPy, OpenCV, Pillow]`

## Getting started

### Prerequisites

- Python 3.9 or later `[FILL: the version you used]`
- pip

### Setup

```
git clone https://github.com/Sindhuja-M-309/waste-classifier.git
cd waste-classifier
pip install -r requirements.txt
```

> Create `requirements.txt` by listing the libraries imported at the top of your scripts, or run `pip freeze > requirements.txt` in your working environment.

### Train the model

```
python train_model.py
```

This trains the model on the images in `data/` and writes `waste_model.keras`.

### Run a prediction

```
python detect_waste.py
```

For the lightweight TensorFlow Lite model:

```
python tflite_infer.py
```

> Add the exact arguments each script expects, for example an image path.

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | `[FILL: %]` |
| Number of classes | `[FILL]` |
| Training images | `[FILL]` |
| Test images | `[FILL]` |

Add a screenshot or short GIF of a prediction here, for example `![Demo](demo.gif)`.

## Dataset

`[FILL: source of the images, such as a public dataset with its link, or images you collected yourself, plus how you split train and test data.]`

## Limitations

- Accuracy depends on lighting, background and image quality.
- The model only recognises the classes it was trained on. Items outside those classes may be misclassified.
- This version identifies waste type only. It does not sort waste physically.

## Future work

- Add sensors to detect waste type automatically at the point of disposal.
- Build a roller mechanism with separate bins to sort waste physically based on the model's prediction.
- Expand the dataset with more item types and real-world photos to improve accuracy.
- Run the TensorFlow Lite model on a small embedded device.

## Author

Developed by **Sindhuja M**, B.Tech Artificial Intelligence & Data Science, Rajalakshmi Institute of Technology. Inspired by the need for sustainable waste management during the 1M1B internship.
