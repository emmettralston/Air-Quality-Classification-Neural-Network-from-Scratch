# Air Quality Classification — Neural Network from Scratch

## Overview
This project builds a multi-layer perceptron (MLP) classifier to label air quality into four categories (Good, Moderate, Poor, Hazardous) using nine environmental and socio-economic features. The entire training pipeline—data cleaning, standardisation, forward and backward passes, and evaluation—is implemented manually with NumPy inside a single Jupyter notebook.

## Repository contents
- `NN_Scratch_AirQuality.ipynb` — step-by-step notebook that downloads the dataset, prepares the data, and trains/evaluates the scratch-built neural network.
- `updated_pollution_dataset.csv` — cached copy of the Kaggle dataset used in the notebook (downloaded automatically if missing).

## Requirements
- Python 3.9+
- Packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `kagglehub`

Install the dependencies with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn kagglehub
```

## Getting started
1. Clone this repository and navigate into it.
2. (Optional) Create a virtual environment for isolation.
3. Launch Jupyter Lab/Notebook and open `NN_Scratch_AirQuality.ipynb`:
   ```bash
   jupyter lab
   ```
4. Run the notebook cells in order. The first cell fetches the latest dataset via `kagglehub`; if you already have `updated_pollution_dataset.csv` locally, the download step is skipped automatically.

## Data & features
Each record contains the following input features:
- Temperature (°C)
- Humidity (%)
- PM2.5 and PM10 concentrations (µg/m³)
- NO₂ and SO₂ concentrations (ppb)
- CO concentration (ppm)
- Proximity to industrial areas (km)
- Population density (people/km²)

The target label is a categorical air-quality rating with four possible classes. The notebook performs validation to ensure these columns exist, normalises label text, and drops rows with missing values before training.

## Notebook workflow highlights
1. **Column standardisation** — renames raw CSV headers to the expected feature names.
2. **Data validation & encoding** — enforces the four-class label scheme and encodes targets as integers.
3. **Train/val/test split** — stratified split (70/15/15) with z-score standardisation using scikit-learn utilities.
4. **Neural network implementation** — defines a 9-64-32-4 MLP with ReLU activations, softmax output, cross-entropy loss, and mini-batch SGD with optional class weighting and early stopping.
5. **Evaluation & diagnostics** — reports accuracy, weighted/unweighted loss, per-class precision/recall/F1, and plots learning curves plus a confusion matrix heatmap.

Feel free to adjust hyperparameters (learning rate, batch size, hidden units, epochs) directly inside the notebook to explore different model behaviours.
