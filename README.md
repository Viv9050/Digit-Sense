# DigitSense

Adaptive Image Feature Extraction and Classification - mini project for the Pattern Matching course.

The app extracts pixel-level and statistical features from handwritten digit images, selects the best K features using SelectKBest (ANOVA F-test), and compares four classifiers.

## Dataset

sklearn `load_digits` - 1797 grayscale images of size 8x8, digits 0 to 9. It comes with scikit-learn, so nothing needs to be downloaded.

## Features

- Pixel features: 64 raw pixel values + 16 bin intensity histogram
- Statistical features: mean, std, variance, skewness, kurtosis, entropy, min, max, row-wise mean, column-wise mean
- Combined: both of the above

## Classifiers

KNN, SVM, Decision Tree, Naive Bayes

Evaluated using accuracy, precision, recall, F1 score and confusion matrix.

## How to run

```
pip install -r requirements.txt
streamlit run main.py
```

Use the sidebar to change the feature type, number of features (K), test size and classifiers. You can also upload your own digit image to see the prediction (white digit on black background works best).
