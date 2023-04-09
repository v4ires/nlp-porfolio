# Spam Busters: Detecting Spam Emails with Machine Learning

## Overview

Spam Busters is a project that aims to detect spam emails using natural language processing (NLP) and machine learning (ML) algorithms. The dataset used for this project is "Spam Email Classification," which is available on Kaggle.

The objective of this project is to create a model that can accurately classify emails as either spam or not spam. The model is built using Python and utilizes NLP techniques to process and classify the emails.

# Dataset

The dataset used for this project is "Spam Email Classification," which is a dataset of SMS messages that are labeled as spam or not spam. The dataset contains 5,572 SMS messages, out of which 747 are spam messages.

The dataset can be found on Kaggle at the following link: https://www.kaggle.com/uciml/sms-spam-collection-dataset.

# Methodology

The spam detection model is built using Python and NLP and ML techniques. The steps involved in the model building process are:

- **Data Preprocessing**: The raw text data is preprocessed to remove any noise and irrelevant information. This involves steps such as removing stop words, converting all text to lowercase, and removing special characters.

- **Feature Extraction**: The preprocessed text data is then converted into a set of numerical features that can be used for classification. In this project, the bag-of-words model is used for feature extraction.

- **Model Building**: A classification model is built using the extracted features. In this project, the model used is a Naive Bayes classifier.

- **Model Evaluation**: The performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Files in Repository

This repository contains the following files:

- **README.md**: A file containing information about the project.
- **data/dataset.csv**: The dataset used for the project.
- **notebooks/1_data_analysis.ipynb**: A Jupyter notebook containing the code with the exploration data analysis over the dataset.
- **notebooks/2_feature_extraction.ipynb**: A Jupyter notebook containing the code for the feature extraction executed over the dataset.
- **notebooks/2_model_building.ipynb**: A Jupyter notebook containing the code for the model building.
- **notebooks/3_model_evaluation.ipynb**: A Jupyter notebook containing the code for the model building.

# Conclusion

TBD