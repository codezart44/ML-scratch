## Linear Regression implemented from scratch
___

- The point of this project is to understand the math behind linear regression. 
- Linear regression finds a best fit line through a set of datapoints and uses that line to evaluate (predict) unseen data. 
- It is assumed that the dataset follows a linear pattern.


### Ordinary Least Squares (OLS)
___
* Find closest approximated solution β' to X•β = y
* Projecting y down on Col X and find a solution for X•β' = ŷ
```
X.T • Xβ' = X.T • y
β' = (X.T • X)^-1 • (X.T • y)
```
//note requires X to have linearly independent columns to be invertible (IMT)


### Gradient Descent (GD)
___
* Minimize MSE (cost function) 
* Calculating pertial derivatives of cost funtion with respect to b & w
```
                      n
MSE = f(b, w) = 1/N * ∑(yi - (w*xi + b))^2
                     i=1

# repeat n times:
            N
db = -2/N * ∑(y - (X•w + b))
           i=1
dw = -2/N * X.T • (y - (X•w + b))

b = b - ɑ * db
w = b - ɑ * dw
```
//note sensitive to learning rate and initial values for bias & weights. Requires scaling.




