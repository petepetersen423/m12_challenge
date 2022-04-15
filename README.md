# Module 12 Challenge

    Pete Petersen
    April 15, 2022


## Background

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, we used various techniques to train and evaluate models with imbalanced classes. We used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## What we created

Using your knowledge of the imbalanced-learn library, we created a logistic regression model to compare two versions of the dataset. First, the original dataset. Second, we resampled the data by using the RandomOverSampler module from the imbalanced-learn library.

For both cases, you’ll get the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Also in this GitHub repository’s we created a credit risk analysis (**report-template.md**)report based on the template provided in the challenges Starter_Code folder.

## Dependancy

In order to run the project, you will need to clone this repository and install the following dependancies:

```

# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from imblearn.metrics import classification_report_imbalanced

```

## Files in this repo

\Resources\lending_data.csv
credit_risk_resampling.ipynb
report_template.md
README.md

Additionally, there are three image files that are used in the analysis report markdown files.

12-1-model-fit-predict-diagram.png
original_confusion.png
resampled_confudion.png


