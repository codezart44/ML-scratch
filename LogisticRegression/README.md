## Logistic Regression
___

## Scaling
Not needed for y data (binary values between 0 and 1)
Beneficial for X data, align features and avoid exploding- / vanishing gradients.

## Derivations

$$
log(n)
$$


sigmoid function of linear function / variable
σ(x) = 1/(1+e^-x)
alt.

ŷ = σ(wx+b) = 1/(1+e^-(wx+b))

Calculating error with cross entropy function:

The cross entropy loss function comes from the MLE of estimating w & b that maximizes the likelihood of observed data. 

Given some input xi the predicted probability p(y=1|xi) for the positive class is given by:
p = p(y=1|xi) = σ(w*x + b) = 1/(1+e^-(wx+b))

### MLE:
for a single observation xi the likelihood of seeing yi is:
li = p^yi * (1-p)^(1-yi)

for N observations X the likelihood of seeing the outcome of y is and each yi ∈ {0, 1}:
    N
L = ∏(p^yi * (1-p)^(1-yi))
   i=1

since we want to find the maximum of this function and d/dx log(f(x)) = f'(x)/f(x) = 0 
    for f'(x) = 0 which is what we seek we use log-likelihood for MLE. 
    It simplifies the differentiation of L and and finding a MLE.
//note log(A * B) = log(A) + log(B)
       log(A^b) = b * log(A)

         N
log(L) = ∑(log(p^yi) + log((1-p)^(1-yi)))
        i=1

         N
log(L) = ∑(yi*log(p) + (1-yi)*log(1-p))       //want to maximize log-likelihood
        i=1
//cross entropy

           N
-log(L) = -∑(yi*log(p) + (1-yi)*log(1-p))     //eqivalent to minimize the negative log-likelihood (loss function)
          i=1
//loss function

The function quantifies how well the predicted values conform to the label values.
Lower values of -log(li) indicates a better model fit to the data (less loss). 
e.g.
yi = 1 and p(y=1|xi) = 0.8      //decent prediction
-log(li) = -(1*log(0.8) + (1-1)*log(1-0.8)) = -log(0.8) ≈ 0.097

or e.g.
yi = 0 and p(y=1|xi) = 0.05     //good prediction
-log(li) = -(0*log(0.05) + (1-0)*log(1-0.05)) = -log(0.95) = 0.022

or e.g.
yi = 1 and p(1|xi) = 0.05       //bad prediction
-log(li) = -(1*log(0.05) + (1-1)*log(1-0.05)) = -log(0.05) = 1.30

We want the negative log likelihood loss to be as close as possible to zero and thus try to minimize the function with respect to b & w. 



### Finding partial differentials df/db & df/dw
Negative Log Likelihood - NLL (loss)
                N
NLL = f(b,w) = -∑(yi*log(pi) + (1-yi)*log(1-pi))
               i=1

yi*log(pi) + (1-yi)*log(1-pi)

yi*log(1/(1+e^-(w*xi+b))) + (1-yi)*log(1-1/(1+e^-(w*xi+b)))

yi*log(1/(1+e^-(w*xi+b))) + (yi-1)*log(e^-(w*xi+b)) - (1-yi)*log(1+e^-(w*xi+b))

-yi*log(1+e^-(w*xi+b)) + (yi-1)*(w*xi+b) - log(1+e^-(w*xi+b)) + yi*log(1+e^-(w*xi+b))

(yi-1)*(w*xi+b) - log(1+e^-(w*xi+b))

                N
NLL = f(b,w) = -∑((yi-1)*(w*xi+b) - log(1+e^-(w*xi+b)))         // fully simplified before differention
               i=1

         N
df/db = -∑((yi-1) + e^-(w*xi+b)/(1+e^-(w*xi+b)))
        i=1

         N
df/dw = -∑xi((yi-1) + e^-(w*xi+b)/(1+e^-(w*xi+b)))
        i=1


## Notes
Why the product of likelihoods and not the sum of likelihoods?
    N
L = ∏(p^yi * (1-p)^(1-yi))
   i=1

Datapoints are not to be evaluated independently (adding) but rather as a dataset.
With multiplication, if one event has a low likelihood it will significantly affect the overall joint likelihood of the whole dataset. 
This creates the need to generalize the model to have similar fit to all datapoints instead of extremely well fit to some and terrible to other.
E.g:
P(A) = 0.5, P(B) = 0.5
Addition:               P(A&B) = 0.5+0.5 = 1.0
Multiplication:         P(A&B) = 0.5*0.5 = 0.25

P(A) = 0.05, P(B) = 0.95
Addition:               P(A&B) = 0.05+0.95 = 1.0
Multiplication:         P(A&B) = 0.05*0.95 < 0.05

With addition large likelihoods can compensate for small likelihoods as seen in the example above where even though the likelihood of A is only 0.05 the large likelihood of B at 0.95 weighs up for that. This causes to model to not generalize well but rather increases variance as fitting well to some datapoints can compoensate for bad fits to other datapoints. That is not the case for multiplication where the maximum joint likelihood is reached when all likelihoods are close to equal. 

Multiplying many small likelihoods together poses other problems however, a small joint likelihood (product) and difficulty differentiating. To compensate for miniscule joint likelihoods we use the log of the product which firstly turns it into a sum (facilitating differentiation) and secondly turns small decimal likelihoods into large negative values. 
         N
log(L) = ∑(log(p^yi) + log((1-p)^(1-yi)))
        i=1

By multiplying by negative one (-1) and dividing by the number of samples the negative logs turns positive and the optimization problem turns into a problem of minmizing the average of the joint negative log-likelihood (NLL) 
                 N
-log(L)/m = -1/m*∑(log(p^yi) + log((1-p)^(1-yi)))
                i=1

This aligns with the standard of machine learning optimization problems which consist of minimizing the loss/cost of an accuracy function of some kind. 