"""
A little script that produces a Fibonacci number
every half second forever until it is interrupted.
"""

import time

# Initialize
a = 0
b = 1

# Make Fibonacci numbers
while True:
    a, b = b, a + b
    print(b)
    time.sleep(0.5)

