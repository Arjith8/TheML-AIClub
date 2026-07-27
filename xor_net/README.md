# Neural Network

## What?
- To understand neural network we need to understand what a neuron is
    - A neuron is in some sense a unit recieves some signal does some conputation and then return out some value.
    - `input -> A(G(input)) -> output`
    - What is G(input)?
        - say its `x @ w + b`
    - What is A()?
        - This is what we call the activation function. What it is will be discussed later.

## Why?
- It was introduced in an attempt to be able to generalize more complex data.
- You might be like, "Hey, assuming there is no activation function since u never explain it I dont see any reason to use this.
I mean say I have 2 neurons with weights [1, 2] and [2, 3] and assuming their result is just summed to get `output`
any `output` I want from this network can just be created from a linear model like in this case with w = [3, 5]"
    - This is such a valid question and this is also why activations are important.
        - So what does activation do in naive terms its a function that introduces some amount of non linearity to the model, i.e., all our 
        models till now relied on a linear decision boundary to separate between say the 2 sides of a binary classification model but what if this line
        was well a curve instead can we model it. NO. This is why activations are used and this without an activation we could say that
        a neural netork is just a over engineered linear regression model
