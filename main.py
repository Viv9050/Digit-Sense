import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from scipy.stats import skew, kurtosis, entropy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# some pixels are always 0 in the dataset, f_classif warns about them
warnings.filterwarnings("ignore")

CLASSIFIERS = ["KNN", "SVM", "Decision Tree", "Naive Bayes"]


def pixel_features(img):
    hist, _ = np.histogram(img, bins=16, range=(0, 16))
    return np.concatenate([img.flatten(), hist])


def stat_features(img):
    flat = img.flatten()
    hist, _ = np.histogram(flat, bins=16, range=(0, 16))
    stats = [flat.mean(), flat.std(), flat.var(), skew(flat), kurtosis(flat),
             entropy(hist + 1e-9), flat.min(), flat.max()]
    feats = np.concatenate([stats, img.mean(axis=1), img.mean(axis=0)])
    return np.nan_to_num(feats)


def get_features(img, ftype):
    if ftype == "Pixel":
        return pixel_features(img)
    if ftype == "Statistical":
        return stat_features(img)
    return np.concatenate([pixel_features(img), stat_features(img)])


def get_model(name):
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=3)
    if name == "SVM":
        return SVC(kernel="rbf")
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    return GaussianNB()


@st.cache_data
def load_data():
    digits = load_digits()
    return digits.images, digits.target


@st.cache_data
def build_features(ftype):
    images, _ = load_data()
    return np.array([get_features(img, ftype) for img in images])


@st.cache_resource
def train_models(ftype, k, test_size, names):
    _, labels = load_data()
    X = build_features(ftype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=42, stratify=labels)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # adaptive part: keep only the K most discriminative features
    selector = SelectKBest(f_classif, k=k).fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    models, preds = {}, {}
    for name in names:
        models[name] = get_model(name).fit(X_train, y_train)
        preds[name] = models[name].predict(X_test)
    return scaler, selector, models, preds, y_test


def prepare_upload(file):
    img = Image.open(file).convert("L")
    # dataset digits are light on a dark background
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    # crop around the digit and center it in a square, like the dataset
    box = img.point(lambda p: 255 if p > 30 else 0).getbbox()
    if box:
        img = img.crop(box)
    side = max(img.size)
    square = Image.new("L", (side, side), 0)
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    arr = np.array(square.resize((8, 8), Image.BOX), dtype=float)
    return arr / max(arr.max(), 1) * 16


st.title("Adaptive Image Feature Extraction and Classification")
st.write("Pattern Matching mini project using the sklearn digits dataset (8x8 images, digits 0-9).")

images, labels = load_data()

st.sidebar.header("Settings")
ftype = st.sidebar.radio("Feature type", ["Pixel", "Statistical", "Combined"])
n_feat = build_features(ftype).shape[1]
k = st.sidebar.slider("Number of best features (K)", 1, n_feat, min(20, n_feat))
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
names = st.sidebar.multiselect("Classifiers", CLASSIFIERS, default=CLASSIFIERS)

st.subheader("Sample Images")
fig, axes = plt.subplots(1, 8, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(f"Label: {labels[i]}")
    ax.axis("off")
st.pyplot(fig)

if not names:
    st.warning("Select at least one classifier from the sidebar.")
    st.stop()

st.write(f"Feature type: {ftype} | Total features: {n_feat} | Selected (K): {k} | Samples: {len(labels)}")
scaler, selector, models, preds, y_test = train_models(ftype, k, test_size, tuple(names))

st.subheader("Classifier Results")
table = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}
for name in names:
    pred = preds[name]
    table["Classifier"].append(name)
    table["Accuracy"].append(round(accuracy_score(y_test, pred), 4))
    table["Precision"].append(round(precision_score(y_test, pred, average="macro", zero_division=0), 4))
    table["Recall"].append(round(recall_score(y_test, pred, average="macro", zero_division=0), 4))
    table["F1 Score"].append(round(f1_score(y_test, pred, average="macro", zero_division=0), 4))
st.dataframe(table, hide_index=True)

best = names[int(np.argmax(table["Accuracy"]))]
st.write(f"Best classifier: {best}")

st.subheader("Accuracy Comparison")
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.barplot(x=table["Classifier"], y=table["Accuracy"], ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
st.pyplot(fig)

st.subheader(f"Confusion Matrix ({best})")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, preds[best]), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Predict Your Own Digit")
file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
if file:
    small = prepare_upload(file)
    feat = get_features(small, ftype).reshape(1, -1)
    feat = selector.transform(scaler.transform(feat))
    result = models[best].predict(feat)[0]

    col1, col2 = st.columns(2)
    col1.image(Image.open(file), caption="Uploaded image", width=200)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(small, cmap="gray")
    ax.axis("off")
    col2.pyplot(fig)
    col2.write("Converted to 8x8")
    st.write(f"Predicted digit ({best}): {result}")
