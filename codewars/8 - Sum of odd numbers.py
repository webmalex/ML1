# http://www.codewars.com/kata/55fd2d567d94ac3bc9000064/train/python
#              1
#           3     5
#        7     9    11
#    13    15    17    19
# 21    23    25    27    29
# ...

# 5 | 21 = (4+3+2+1)*2+1

for i in range(1,6):
    s = sum(range(1,i))*2+1
    print(s, sum(range(s, s+i*2, 2)))

from Test import *

row_sum_odd_numbers = lambda n: (lambda s: sum(range(s, s+n*2, 2)))(sum(range(1, n))*2+1)

test.assert_equals(row_sum_odd_numbers(1), 1)
test.assert_equals(row_sum_odd_numbers(2), 8)
test.assert_equals(row_sum_odd_numbers(13), 2197)
test.assert_equals(row_sum_odd_numbers(19), 6859)
test.assert_equals(row_sum_odd_numbers(41), 68921)