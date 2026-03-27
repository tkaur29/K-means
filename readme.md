# K-Means Customer Segmentation Demo 🎯

Interactive Gradio application for visualizing K-Means cluster assignment on 2D customer-like data.

[Live Demo](https://tanveerkaur29-kmeans-visualizer.hf.space/) (Hugging Face)

---

## Overview

Enter an `(x, y)` point and the app assigns it to the nearest cluster, then plots the result alongside the dataset and cluster centers.

## Features

- Loads a pre-trained model from `kmeans_custom.pkl`
- Displays dataset scatter and cluster centers
- Predicts cluster ID for custom input values
- Visualizes the selected point with the predicted cluster

## Requirements

| Item | Version |
|------|---------|
| Python | 3.9+ |
| Dependencies | `requirements.txt` |

## Quick Start (To run locally)

1. Install dependencies.

```bash
pip install -r requirements.txt
```
2. Ensure `kmeans_custom.pkl` is present in the project root.
3. Run the app.

```bash
python app.py
```
4. Open the local Gradio URL shown in your terminal.

## Project Files

- `app.py` - Gradio UI and prediction/plot logic
- `requirements.txt` - Python package dependencies
- `kmeans_custom.pkl` - Saved cluster model

