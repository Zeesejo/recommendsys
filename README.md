# Personalized Career Pathway Recommendation System

This repository contains the code and documentation for a personalized career pathway recommendation system using machine learning and deep learning techniques. The system utilizes a dataset of resume summaries categorized into various job roles and explores different models to provide personalized and accurate career recommendations.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Support Vector Machines (SVM)](#support-vector-machines-svm)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest](#random-forest)
  - [DistilBERT](#distilbert)
  - [TF-IDF and Cosine Similarity](#tf-idf-and-cosine-similarity)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In today's competitive job market, finding the right career path is crucial for individuals seeking professional growth. This project presents a practical approach to developing a career pathway recommendation system using machine learning and deep learning techniques. The system utilizes a dataset of resume summaries categorized into 28 job categories and explores various models to provide personalized and accurate career recommendations.

## Dataset

The dataset used in this study is the `Updated_Resume_Dataset.csv`, which consists of three columns: labels, category (28 different job categories), and resume summary. This dataset provides a comprehensive collection of resume summaries that are categorized into various job roles, making it suitable for developing a career pathway recommendation system.

## Methodology

The methodology involves several preprocessing steps, including text cleaning, tokenization, removal of stop words, lemmatization, and feature extraction using TF-IDF. The preprocessed data is then used to train and evaluate various machine learning and deep learning models.

## Models

### K-Nearest Neighbors (KNN)

The KNN model is trained using the TF-IDF transformed data. The code snippet for training and evaluating the KNN model can be found in the repository.

### Support Vector Machines (SVM)

The SVM model is trained using the TF-IDF transformed data. The code snippet for training and evaluating the SVM model can be found in the repository.

### Logistic Regression

The Logistic Regression model is trained using the TF-IDF transformed data. The code snippet for training and evaluating the Logistic Regression model can be found in the repository.

### Random Forest

The Random Forest model is trained using the TF-IDF transformed data. The code snippet for training and evaluating the Random Forest model can be found in the repository.

### DistilBERT

The DistilBERT model is trained using the preprocessed resume data. The code snippet for training and evaluating the DistilBERT model can be found in the repository.

### TF-IDF and Cosine Similarity

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert textual data into numerical features. Cosine similarity is then used to find the most similar resumes. The code snippet for performing TF-IDF transformation and using cosine similarity can be found in the repository.

## Evaluation

The performance of the models is evaluated using accuracy, precision, recall, and F1-score. Confusion matrices are also plotted to visualize the performance of the models.

## Usage

To use the recommendation system, follow these steps:

1. Clone the repository:
   git clone https://github.com/Zeesejo/recommendsys.git
   cd recommendsys
