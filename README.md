Project:

To predict the Credit limit for Customers.

Data Set Information: This research aimed at the case of customers default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. Because the real probability of default is unknown, this study presented the novel Sorting Smoothing Method to estimate the real probability of default. With the real probability of default as the response variable (Y), and the predictive probability of default as the independent variable (X), the simple linear regression result (Y = A + BX) shows that the forecasting model produced by artificial neural network has the highest coefficient of determination; its regression intercept (A) is close to zero, and regression coefficient (B) to one. Therefore, among the six data mining techniques, artificial neural network is the only one that can accurately estimate the real probability of default.

Attribute Information:

{NOTE: The following is updated information from the source’s author} This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:

X1: Credit limit(NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.

X2: Gender , male or female

X3: Education (graduate school, university, high school or others).

X4: Marital status (1 = married; 2 = single; 3 = divorce; 0=others).

X5: Age (year).

X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows:

X6 = the repayment status in September, 2005;

X7 = the repayment status in August, 2005;

X8 = the repayment status in July 2005;

X9 = the repayment status in June 2005;

X10 = the repayment status in May 2005;

X11 = the repayment status in April 2005;

The measurement scale for the repayment status is: -2: No consumption; -1: Paid in full; 0: The use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above . X12-X17: Amount of bill statement (NT dollar).

X12 = amount of bill statement in September, 2005;

X13 = amount of bill statement in August, 2005;

X14 = amount of bill statement in July, 2005;

X15 = amount of bill statement in June, 2005;

X16 = amount of bill statement in May, 2005;

X17 = amount of bill statement in April, 2005.

X18-X23: Amount of previous payment (NT dollar).

X18 = amount paid in September, 2005;

X19 = amount paid in August, 2005;

X20 = amount paid in August, 2005;

X21= amount paid in August, 2005;

X22 = amount paid in August, 2005;

X23 = amount paid in April, 2005.

Y: client's behavior; Y= default/not default
