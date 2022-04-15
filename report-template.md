# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.

Using imbalanced-learn and scikit-learn libraries,  evaluate two machine learning models by using resampling to determine which is better at predicting credit risk. First, we used the orginal dataset to model and fit the data.  Next, we  use the oversampling **RandomOverSampler** from **Scikit.learn** to create and train another model. Lastly, we view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

* Explain what financial information the data was on, and what you needed to predict.

For this excerside we use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.  The provided data includes loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks,and  total_debt.  The dataset also includes our target variable **Loan Status'**.  Since the labeled **'Loan Status'** contains either a value of 0  meaning that the loan is healthy, or a value of 1 meaning that the loan has a high risk of defaulting.  We will select a model that minimizes the false positives (incorrectly predicted healthy loans)in the 0 class and the false negatives(loans the we failed to predict as unhealthly) in the 1 class.  However, we belive that we should seek to minimize the incorrectly predicted healthy loans because because these loans will not be priced appropriately to handle the probability of default, while a false prrediction in the unhealthy class is not a risk to the lender, but only falsely punishes the borrower since they will be saddled with an in appropriate rate compensate for teh default risk.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

Let's examine the the labeled data.  First we check the balance of values in the original **target**.  We take note that the dataset is very unbalnced at 97% healthy and 3% uhealthy.  

```
# Check the balance of our target values
y.value_counts()
```

    0    75036
    1     2500  
    Name: loan_status

We therefore determine that the model will benefit by training with resampled data in order to balance the minority class to the majority.  In order to leave the orginal targets unmolested, we first split the data into test and training and observe the following counts in our test and training y variables.

```
# Count the distinct values of the resampled labels data
display(y_train.value_counts())



# Import the RandomOverSampler module form imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

# Instantiate the random oversampler model
# # Assign a random_state parameter of 1 to the model
random_oversampler = RandomOverSampler(random_state=1)

# Fit the original training data to the random_oversampler model
X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)

display(y_resampled.value_counts()
y_resampled.value_counts()
```
    0    56271
    1     1881  
    Name: loan_status, dtype: int64

    0    56271
    1    56271  
    Name: loan_status, dtype: int64

Initally we verify the the test train split presevered the 3% minority class.  Next, We note that after resampling the original minority class is now appropriately in balance with the majority class.

* Describe the stages of the machine learning process you went through as part of this analysis.

    ### 1.  Model
    A machine learning model mathematically represents something in the real world. A model starts as untrained. That is, we haven’t yet adjusted it to make sense of the data. You can think of an untrained model as a mathematical ball of clay that’s ready to be shaped to the data.

    ### 2.  Fit
    The fit stage (also known as the training stage) is when we fit the model to the data. In the mathematical ball-of-clay analogy, we adjust the model so that it matches patterns in the data. Recall that in our time series forecasting, the Prophet tool built a model that matched the time series components of the data. We could then use that model to forecast the values that future data might have. The fit stage of supervised learning works the same way. This is when the model starts to learn how to adjust (or train) itself to make predictions matching the data that we give it.

    ### 3.  Predict
    Once the model has been fit to the data (that is, trained), we can use the trained model to predict new data. If we give the model new data that’s similar enough to the data that it’s gotten before, it can guess (or predict) the outcome for that data.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

```
# Print the classification report for the model
print(classification_report_imbalanced(y_test, y_original_pred))
```


                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
```
# Print the classification report for the model

print(classification_report_imbalanced(y_test, y_resampled_pred))
```

                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619



## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
