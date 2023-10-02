## Linear Regression implemented from scratch
___

### General
___
- This is an algorithm focused implementation of Linear Regression for ML applications
- The point of this project is to gain a deeper undestanding of the math behind linReg. 
- Linear regression finds a best fit line through a set of datapoints and uses that line to evaluate (predict) unseen data. 


### Assumption
___
- Dataset follows linear pattern

### Goal
___
- Find slope (trend) of data

### Fomulas
___
```
y_hat = w*x + b         # weight (w) & bias (b)

                      n                       n
MSE = J(w, b) = 1/N * ∑(yi - y_hat)^2 = 1/N * ∑(yi - (w*xi + b))^2
                     i=1                     i=1
```

### Approach
___
- Trying to minimize MSE to have find best fit
- Gradient Descent

### Gradient Descent
___
```python
# Descent as in tuning w & b to minimize MSE as fast as possible

w = w - ɑ*dw            # (delta w)
b = b - ɑ*db            # (delta b)

```

