# Fraud-Credit-Card-Detection-Using-Machine-Learning

<h1 align="left"> Abstract </h1>

<p align="left"> 
Credit card fraud is a significant problem, with billions of dollars lost each year. Machine learning can be used to detect credit card fraud by identifying patterns that are indicative of fraudulent transactions. Credit card fraud refers to the physical loss of a credit card or the loss of sensitive credit card information. Many machinelearning algorithms can be used for detection. This project proposes to develop a machine-learning model to detect credit card fraud. The model will be trained on a dataset of historical credit card transactions and evaluated on a holdout dataset of unseen transactions.

</p>

<h1 align="left"> About Dataset </h1>
<p align="left">
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
</p>
<h1 align="left"> Data Source </h1>
<p align="left"> 
  The dataset will be sourced from Kaggle, specifically the "Credit Card Fraud Detection" dataset.This dataset contains anonymized transaction details, including time, amount, and class labels (fraudulent or legitimate), with over 284,000 transactions. 
  
<b> Kaggle Dataset: </b>
<a href="https://www.kaggle.com/datasets/isaikumar/creditcardfraud/data"> Credit Card Fraud Detection Dataset </a>
</p>
  
<h1 align="left"> Algorithms Used </h1>

<div align="left">
  
<a align="left"> 1. Logistic Regression (LR) </a>

 <a align="left"> 2. Decision Tree (DT) </a>

<a align="left"> 3. Logistic Regression (RF) </a>

<a align="left"> 4. Support Vector Machine (SVM) </a>

</div>

<h2> Evaluation Matrices </h2>
<table> 
  <tr>
    <th> Report </th>
    <th> 0,1 </th>
    <th> Accuracy	</th>
    <th> Precision </th>
    <th> Recall	</th>
    <th> F1 score </th>
  </tr>
  
  <tr>
    <td> <b>Decision Tree</b> </td>
    <td> 0
  
1 </td>
    <td> 1.00

1.00 </td>
    <td> 1.00
  
1.00 </td>
    <td> 1.00

1.00 </td>
    <td> 1.00

1.00 </td>
  </tr>

  
  <tr>
    <td> <b>Random Forest</b> </td>
    <td> 0

  1 </td>
    <td>1.00

1.00 </td>
    <td> 1.00

1.00 </td>
    <td> 1.00

1.00 </td>
     <td> 1.00

1.00 </td>
  </tr>

  
  <tr>
    <td> <b>Support Vector Machine</b> </td>
         <td>0
      

1</td>
     <td> 1.00

1.001 </td>
     <td> 1.00

1.00 </td>
     <td> 1.00

1.00 </td>
     <td> 1.00

1.00 </td>
  </tr>
  
  <tr>
    <td> <b>Hybrid Approach</b> </td>
    <td>0
    
  1</td>
    <td> 1.00

1.00 </td>
     <td> 1.00

1.00 </td>
     <td> 1.00

1.00 </td>
     <td> 1.00

1.00 </td>
  </tr>
</table>  

<h1 align="left"> Conclusion And Future Work </h1>

<p align="left"> 
Credit card fraud detection is crucial for financial security, and machine learning has significantly improved its accuracy. By analyzing transaction patterns, ML models can identify fraudulent activities and reduce financial losses. However, challenges such as imbalanced datasets, false positives, and evolving fraud tactics require continuous Credit card fraud detection is a crucial aspect of financial security, helping banks and businesses identify fraudulent transactions and prevent financial losses. Machine learning plays a significant role in modern fraud detection systems by analysing transaction patterns and detecting anomalies.
</p>

