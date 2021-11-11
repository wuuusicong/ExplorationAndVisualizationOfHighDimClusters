import random
import numpy as np
initLabel = [i for i in range(4)]
testLabel = [3,4,5,6]
digit_to_label = zip(testLabel, initLabel)
print(dict(digit_to_label))
RANDOM = 42
np.random.seed(RANDOM)
print(RANDOM)
random.seed(RANDOM)
print(RANDOM)

