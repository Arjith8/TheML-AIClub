# Logistic regression
## Why?
* Suppose we want to classify things such as spam, where each email could be one of 2 classes spam or not spam. If we were to use
linear regression then we would get number outputs say something like 20 for 1 or 1 for 0 but for us it would be easier if
we got the probability of something being a spam or not spam instead
* This is where logisting regression comes in,
	* so what does it do
		* similar to liner regression it computes `z = wx + b` but instead of this being the output its output is passed through
		  a sigmoid function `1/(1+e^-z)` this function will ensure that we get a result between 0 and 1
## How do we train?
* Say we have 2 classes 0 and 1 and for each training sample if the true label is 1, we want the model to predict a probability close to 1. If the true label is 0, we want it to predict a probability close to 0.
* Lets say we decide to train it the same way as linear regression ie we predict and then we check error using MSE
	* so the derivation will be 
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}
$$
$$
\frac{\partial L}{\partial \hat{y}} = 2(\hat{y}-y)
$$
$$
\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})
$$
$$
\frac{\partial z}{\partial W} = X
$$
$$
\frac{\partial L}{\partial W} = 2(\hat{y}-y).\hat{y}(1-\hat{y}).X
$$
* Now lets say I predicted P(1) for a specific data point as 0.1, then
$$
\frac{\partial L}{\partial W} = 2*(0.1-1).(.9)(.1).X
\space
\frac{\partial L}{\partial W} = -0.162X

$$
* Here we can see that eventhough the probabilites are very different in the end the gradient only moves a small bit
* This is where **Binary Cross Entropy (BCE) **loss** **comes into play.
$$
L= -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)]
$$
$$
\frac{\partial L}{\partial \hat{y}} = - \frac{y}{\hat{y}}+\frac{1-y}{1-\hat{y}}
$$
$$
\frac{\partial L}{\partial W} = -10.(.9)(.1).X
\space
\frac{\partial L}{\partial W} = -.9X
$$

