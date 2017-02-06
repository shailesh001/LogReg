import numpy as np

w = 20

# How close can we get to zero
for i in range(50):
    w = w - 0.1*2*w
    print(w)

