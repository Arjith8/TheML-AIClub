# Thoughts

## First Goal: How do I update my neural netwroks such that I can implement batch norm
- Why this question? 
    - I have ChatGPT giving me 1 topic every day and this was one of them

- Why I thought my current API design will be not great
    - I was considering how I would implement this I had a few questions?
        - What is Batch Normalization?
            - Core idea: We ran our data through some layer(here data could be output from other hidden layers) we get some activation values
                - Now my first thought was say this was very low like 0.0001 say I mean the next say NN layer will learn w such that this will be higher as in the next layer activation values could be higher so this cant be the reason
                - Then I was like what all things could this activation affect, its obvioud it will affect the forward pass but what about backward
                    - ys it does, its kinda easy to derive so not gonna go into it
                - But if activation value is small then that means the neuron is potentially not that imp for the specific data point right
                    - But then what if say I got activation 0.0001 but my weights 3000x it then I would claim that neuron is imp
                    - mmmm but still I could use batch norm ig say all activations in a specific layer is small or something like 0.001 they have similar magnitude or something then sure I think it could be useful
                    - but we arent checking scale as in hey this activation is 0.001 all the way around in this layer so lets normalize it NO so why?
            - Honestly I dont have an answer tho this even after thinking for a while, the reason Gpt5.4 gave is `BatchNorm makes the optimization problem better conditioned.` I mean ohk but I still dont see why

- Either way I do need better API design I mean I cant just write w1, w2, w3 for each linear layer I am adding so lets start with that.
    - I am like hey lets create a linear class
        - Fine what will it do?
            - Something like it takes some input feature dim then do the wTx+b then send the resp to next layer
            - But if I only take input dim there is no way to say have a new layer where output dim is say n - 10 where n is the input dim
            - ie if I dont do this then say I have a linear layer all will have to be same dim?
                - Ya I cant have that I mean I cant change the dimensions at say the activation or norm layer as thats not what they are for

    - So `Class Linear` which has 2 member vars related to input and output dim set by users rest ig we will do it ourselves
        - But I want multiple layers to be able to pass infomation from one layer to another

- Ig as a base case say my NN is only one linear layer and say thats it, then how do i do this?
    - Say we have some class `Linear` then it will have the linear specific params such as w and b but how do i pass data to it
    - I mean I could do something like hey `Linear(x)` sorta way but what if I have more layers, then I will have to do something like
```
layer1_activations = Linear(x)
```
- This would be a bit tedious, I have to take results then each layer will have to at some point be added to variables since as it grows we will have something like
```
Sigmoid(Linear(Relu(Linear(x))))
```
- I kinda like this syntax of writing, but what would be its cons
    - I can assume that data will fall through properly, we just need a data param
    - But how will I keep known from one layer to another. Mmmm 5 min of thought and this is a bad idea
        1. I have to keep track of `w`s and `b`s which since its a function will go outa scope so will have to return it.
            - Or I will have to store it in a some store that acts as a data store but then every layer is heaviliy dependent on one source mmm I dont like the idea

- Since I wouldve needed a data store in the first place mught as well revert back to using classes
    - So as before we have `class Linear` say if we were to take inspiration from pytorch I will need some syntax where I can easily switch between non gradient requiring and gradient requiring layer
- How should I do this, I will need some parent class which will act as a intermediatery between layers, I dont want to have to move activations around manually



## Things I noticed along the way
- When I first implemented Linear class I thought I will use rand instead of randn because even in the other ones when I asked queries I just always used randn so I thought it was standard.
    - Now that I have checked the kinda output both gives ig I understand why I will stick with randn
        - rand only provides +ve values whereas randn also provives -ve values both are in the fractions range ie they provide < abs(1) values
        - I was like why would I need -ve values say all my data points are well +ve and my w is also +ve then my output is positive then if I use relu as activation is there any use nah
        - But I could use a diff activation so what could be the real reason
            - Say I am building ML model for banking, say loan underwriting then
                - will all the data have +ve impact on whether the user is loan worth no. Fine but even then cant I go from a +ve num to -ve? Yes
                - Except for it taking longer currently I cant seem to find any reasons as to why its imp tho.
