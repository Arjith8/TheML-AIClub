# Linear Regression

- One of the earliest models in ML
- Tries to model the data in a linear fashion.

## Problem it solved

- Lets say we are trying to predict whether the lisitng price of a property is representative of past trends
- There would be 10s if not 100s of data points that could give you an idea as to how much it could be.
- For a human being maybe taking a few of these data points and predicted a range of value could be possible based on
expertise but using all the data we have might not be possible. 
- This is where linear regression comes, bridges this gap in human ability by mathematically modelling a line such that 
it closely resembles the trends in the data and how it affects listing price

## Math
```
y = wx + b
```
- This is the base for linear regression. Where `w` and `b` are model parameters learned over the training session.
- What are we trying to achive? 
    - Learn `w` and `b` such that error, some method to quantify the difference between actual value and predicted value, is minimized.
