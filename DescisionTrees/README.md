# Decision Trees

## Terminology
- __Root Node__: First node / decision. 
- __Branch / Internal Node__: A decision rule that splits the data. 
- __Leaf / Terminal Node__: End nodes that represent the classification.
- __Decision Boundaries__: Division made in the feature space based on the decision rules. 
- __Depth__: Longest path from root to a leaf.
- __Pruning__: Removing branches (sections) of the tree that provide little predictive power. *_Occams Razor_ 
- __Entropy__: Measure of impurity - randomness or chaos. High entropy for even distribution, low entropy for uneven distribution. 
- __Gini Impurity__: Another measure for impurity, especially used for CART algorithm for a branch.
- __Information Gain__: Difference in entropy / gini impurity before and after decision split.
- __Feature Importance__: Weight of feature for making a prediction. Which feature is most important? 


## Notes
Decision trees build on the concept of entropy - "unpredicability". The more chaos or unorder in an eventspace the less predictable outcomes become - and thus the higher the entropy becomes. 

Decision trees try to minimize the entropy (unorder) in a dataset - it does so by sorting (separating) the class labels into groups. The algorithm works by splitting the dataset by different features (X) and thresholds to partition the labels (classes). The threshold split that yields the best separation of classes is always prioritized since this minimizes the entropy of the outcome. With perfect separation a state of zero entropy is reached (no mixture of classes and thus perfect order). 

Before we move onto an example, we need to cover some fundamentals of information and entropy.

Entropy is the expected information from a state (with some event space).
Information is measured through the probability of an event $P(E)$. Information is measured in bits - One bit equals the information gained from the outcome of a uniform binary event space (0 or 1).

### Shannon Information

$X \in \{0, 1\}$

$P(X=1) = -log_2(1/2) = log_2(2) = 1 \text{ bits}$

Shannon Information from a die toss. Each outcome has a certain probability and less likely events have higher infromation content. Seeing the obvious provides little to no information... The die toss outcome is modeled as a discrete random variable (rv) X following a uniform distribution (fair die).

$X \sim \mathcal{U}(1,6) = \frac{1}{6-1+1} = \frac{1}{6}$

$\Omega = \{1, 2, 3, 4, 5, 6\}$

$p_{any} = P(X = \text{Any Value}) = 1$

$$
\begin{align}
I(p_{any}) &= -log_2(p_{any}) \\
&= -log_2(1) \\
&= 0 \\
\end{align}
$$

We are guaranteed to get some number so there is no uncertainty in the event of seeing any number from a die toss. Thus the information gained on seeing any one number is 0 from a die toss. 

$p_{even} = P(X = Even) = \frac{3}{6} = \frac{1}{2}$

$$
\begin{align}
I(p_{even}) &= -log_2(p_{even}) \\
&= -log_2(\frac{1}{2}) \\
&= log_2(2) \\
&= 1 \text{ bits}
\end{align}
$$

As we recall from earlier, the information gained from the outcome of a uniform binary event space is exactly 1 bit! And in this case it is binary by each outcome either being _even_ or _odd_ with the same probability. 

$p_{two} = P(X = 2) = \frac{1}{6}$

$$
\begin{align}
I(p_{two}) &= -log_2(p_{two}) \\
&= -log_2(\frac{1}{6}) \\
&= log_2(6) \\
&\approx 2.58496 \text{ bits}
\end{align}
$$

We see that the infromation has an inverse relation to the probability of the outcome - the more unlikely the event is the higher information content it holds. Rolling one specific number (2 in this case) is less likely than rolling any of the even numbers {2, 4, 6}, hence the higher information content. 

Generally the Shannon Information formula is given by:

$\Omega - \text{Event Space}$

$\omega - \text{Event}$

$\omega \in \Omega$

$p_{\omega} = P(X \in \omega)$

$$\text{Information} = I(p_{\omega}) = -log_2(p_{\omega}) = log_2(\frac{1}{p_{\omega}})$$




### Entropy
Entropy can be regarded as the expected information content from all possible outcomes in a given state. Considering all possible outcome probabilities we calculate each associated information content and then sum them weighted by their corresponding probabilities. 

Entropy is often denoted as $H$. It measures the expected armount of infirmation that can be seen from a state given some actions.

We have a standard deck of 52 playingcards. 

Let's say we want to know the entropy for drawing a specific suit. There are four in total: Spades, Clubs, Hearts, Diamonds

$X \sim U(1, 4) = \frac{1}{4-1+1} = \frac{1}{4}$

$$
\begin{align*}
H_{suits} &= -\sum_{suits} p(X=suit) \times \log_2(p(X=suit)) \\
&= -\sum_{i=1}^4 \frac{1}{4} \times \log_2(\frac{1}{4}) \\
&= -4 \times \frac{1}{4} \times \log_2(\frac{1}{4}) \\
&= \log_2(4) \\
&= 2
\end{align*}
$$

There is a pattern for when the distribution is uniform, allowing us to simplify to $\log_2(p)$ in the end. The only aspect affecting the final entropy when the distribution is uniform is the probability of each outcome itself, meaning for a 52 playing card deck where we compute the entropy for drawing a specific value is instead $H_{value} = \log_2(13) \approx 3.70$.

A more interesting case (and perhaps more relevant to the topic of Decision Trees) would be to understand how to compute the entropy for a given dataset. The entropy calculated is in relation to the classes (labels) - the more evenly distributed the labels are the 




### Information Gain

### Gini Impurity





## Questions
1. Q: __Are they binary trees?__
- Most decision trees are binary trees as each node has exactly two outgoing edges. Categorical features are split into groups and numerical by some threshold. For algorithms like CART (Classification and Regression Trees) the outcome (groups) are binary.     

2. Q: __How is the optimal threshold calculated for splitting numerical features?__
- ...


