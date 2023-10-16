import pandas as pd 
import numpy as np

inputs = [0 ,2 ,-1 ,3.3 ,-2.7  , 1.1 , 2.2 ,-100]
output =[]

# this is how the RELU activation function work
for i in inputs :
    if i > 0 :
        output.append(i)
    if i <= 0 :
        output.append(0)

print(output)


