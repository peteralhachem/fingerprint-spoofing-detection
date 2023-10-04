
# Fingerprint Spoofing Detection

The goal of the project is the development of a classifier to detect whether a fingerprint image represents
an authentic fingerprint or a spoofed fingerprint, i.e. a fingerprint image obtained by cloning, possibly
with different methods, the fingerprint of an individual. The fingerprint images are represented by
means of embeddings, i.e. low-dimensional representations of images obtained by mapping images to a
low-dimensional manifold (typically few hundred dimensions). To keep the model tractable, the dataset
consists of synthetic data, and embeddings have significantly lower dimension than in real use-cases.
The embeddings are 10-dimensional, continuous-valued vectors, belonging to either the _authentic fingerprint_ (label 1) or the _spoofed fingerprint_ (label 0) class. The embedding components do not have
a physical interpretation.

## Authors

- [@mariomastrandrea](https://github.com/mariomastrandrea)
- [@peteralhachem](https://github.com/peteralhachem)

## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
![Static Badge](https://img.shields.io/badge/university-polito-green)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%233F4F75.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)



## Repository Structure

- `data/` =>  it contains the Fingerprint datasets used for the purpose of the project:
*training set* (used for training and validating the models) and *test set* (used to evaluate the models over unseen data)
- `output/`  =>  it contains all the **cached results*** computed by all the scripts in a JSON format, grouped by
the corresponding section of the final report
  - `4_model_selection/`
  - `5_models_comparison/`
  - `6_model_calibration`
  - `7_model_evaluation`
- `src/`  =>  it contains all the source code of the project
  - `classifiers` => it contains all the types of (binary) **classifiers** employed in the project; 
  each one is a class extending the BinaryClassifier base class
  - `cross_validation` => it contains the CrossValidator and its result class, used to perform **Cross Validation**
  - `evaluation` => it contains the BinaryEvaluator and its result class, used to perform **Evaluation**
  - `features_analysis` => it contains the base functions for all the plots of the **features analysis** phase
  - `preprocessing` => it contains all the **data preprocessing methods** employed by the classifiers, in the form of classes
  - `results` => it contains all the **scripts** to reproduce all the results and plots showed in the *final report*, 
grouped by the corresponding **section** of the report (here you find all the *main* scripts to visualize the results)
    - `1_introduction`
    - `2_task`
    - `3_features_analysis`
    - `4_model_selection`
    - `5_models_comparison`
    - `6_model_calibration`
    - `7_model_evaluation`
  - `score_calibration` => it contains the Binary Logistic Regression calibrator employed for **score calibration**
  - `utilities` => it contains useful functions, used throughout all the code, grouped by scope

*each script will save its results inside the `output/` folder, 
and it will then read them: in each script you can thus skip the computation phase
and just leave the code reading the results, in order to visualize them.
Note: scripts have already the computation part commented (just 1 line of code in the 'main'),
if you want to re-compute the script's result, just decomment the corresponding line
## Documents

- [Introduction](https://github.com/MP-MLPR/fingerprint-spoofing-detection/blob/main/FingerprintSpoofingDetection_2023.pdf) - file where you can find a detailed explanayiom about the task in hand, the dataset, an approach on how to solve
- [final report](https://github.com/MP-MLPR/fingerprint-spoofing-detection/blob/main/302219_293885_final_report.pdf) - A deep analysis of the various techniques used for classification with results associated.
## License

[MIT](https://choosealicense.com/licenses/mit/)

