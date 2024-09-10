# Neural-Networks-in-Python

## Goal: 
Using data about workers, make an AI-based binary classifier that can determine if they work undeclared.
There are 3 types of undeclared workers: without contract, with envelope wage or as a false independent worker.

## Data:
+ <strong>The data </strong>  used is from the 2019 Eurobarometer about undeclared work and from Eurostat.
+ <strong>Selected demographic characteristics </strong>  are age, nationality, satisfaction in life, the frequency of debates on political issues at the national/European/local levels, occupation, emigrant status, etc.
+ <strong>Selected variables related to undeclared work</strong> are the number of acquaintances who have an undeclared job, anticipated sanctions for undeclared work, assessing the risk of being detected in one's own country, and confidence in the authorities regarding the fight against undeclared work, etc.
+ <strong>30 macroeconomic variables </strong> refer to the following categories: poverty and social exclusion, housing and living conditions, health, education, digital skills, gender equality, economic and labor, safety and justice, corruption and trust, assistance and taxes, and government debt.

## Methodology:
+ <strong>Preprocessing </strong> of data, to transform qualitative variable into dummy ones. Dataset size: 27565 x 129
+ <strong>Class imbalance </strong> has to be worked around through undersampling, as oversampling generates very bad results and low test accuracy.
+ <strong>Test and Training sets </strong> are formed respecting the proportions 80%-20%.
+ <strong>Standardization </strong> is applied in the case of MLP and Convolutional networks.
+ <strong>The search for the best model </strong> is done using loops for MLP and Convolutional networks, and GridSearches for Tabnet and Random Forest.
+ <strong>For generalization</strong>, the number of epochs is 5, batch_size is 1 and 20% dropout layers are included.
+ <strong>Storage of models </strong> is done in a dataframe.
+ <strong>Visualization </strong> of ROC curves and heatmaps of the confusion matrix for the best models.
+ <strong>SHAP </strong> values can also be calculated to estimate the importance of variables for the final prediction.
  
 <img src="https://github.com/user-attachments/assets/b20f83a6-9b4a-494f-8098-38d4130c8ff1" alt="Methodology" width="700">
 <br>

## Performance metrics for classification of:

<strong>Undeclared work</strong> (2973 with class 1)  
<img src="https://github.com/user-attachments/assets/e480bb49-dfaf-4521-a5ea-137e1d649e4d" alt="Undeclared Work" width="700">

<strong>Envelope wages</strong> (666 with class 1)  
<img src="https://github.com/user-attachments/assets/091cffd4-917f-4de5-b3dd-7d88ec8ca515" alt="Envelope Wages" width="700">

<strong>False Independent workers</strong> (61 with class 1)  
<img src="https://github.com/user-attachments/assets/808eee62-7d85-4af5-942b-4d1b74566a14" alt="False Independent Workers" width="700">


## Some conclusions:
+ <strong> A higher accuracy</strong>  can be noted  for the MLP and CNN models.
+ <strong> TabNet: </strong> when used with many observations it achieves roughly equal performance to the other algorithms. On the other hand, the smaller the data set used, the lower the performance of the algorithm.
![image](https://github.com/user-attachments/assets/b2cd8e16-b498-4397-b27d-92e9cb7fb323)

SHAP values can offer some insights into the importance of the variables. Among the variables of importance for all 3 forms of informality, work in the service sector and lack of a life partner (which increase the probability of informal behavior), as well as retirement and continuing education (which decrease the probability of participation in the informal economy) stand out.
