# Portfolio 01 — NumPy Preprocessing From Scratch

A foundational Python/NumPy coding exercise developed during KOAI / IOAI 2027 preparation. This project is **not** a predictive ML benchmark or a finished competition entry.

## Objective

Implement and validate four common preprocessing skills without using an off-the-shelf scaler:

1. **Mean centering** per feature using NumPy broadcasting.
2. **Standardization** per feature using population standard deviation (`ddof=0`).
3. **Zero-variance handling** using safe scale values, leaving centered constant columns equal to zero.
4. **Leakage-free fitting and transforming**, learning statistics only from the training split and reusing them for held-out data.

## Structure

- `01-ml-preprocessing/preprocessing.ipynb` — annotated code, small reproducible datasets, tests, explanations, limitations.

## Reproduce

1. Open `01-ml-preprocessing/preprocessing.ipynb` in Google Colab or Jupyter with Python 3 and NumPy installed.
2. Choose **Runtime → Run all** (or **Kernel → Restart & Run All**) to execute top to bottom.
3. The notebook should complete without errors, including the final integrated tests and constant-feature tests.

Dependencies: Python 3, NumPy, notebook environment. No datasets to download, random seeds, or GPU needed.

## Results

- Original training features standardize to mean approximately 0 and population standard deviation approximately 1.
- Held-out test values are scaled using the **training** means and standard deviations, and therefore their column means need not equal 0.
- A constant training-age feature is represented as zero after centering and safe scaling.
- Assert-based checks confirm finite values, expected shapes, mean and standard-deviation properties, transformations using stored training statistics, and behavior on a constant-feature example.

## Limitations

The exercise uses tiny synthetic datasets. The functions assume a numeric NumPy matrix with a compatible number of columns and do not handle missing values, strings, categorical features, outliers, shape errors, or a full predictive model. The assertions are focused checks, not comprehensive test coverage. Further work: compare with `scikit-learn`'s `StandardScaler`, add input validation and missing-value handling, test on unseen realistic data, and evaluate a model using cross-validation.

## Learning note

Developed through guided practice and debugging, with code and explanations iteratively tested in Colab. The work demonstrates foundational preprocessing and experimentation, not independent research-level contribution.