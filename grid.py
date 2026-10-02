import numpy

num_x = 100
num_y = 100
x = numpy.zeros([num_x+1, num_y+1])
xi = numpy.linspace(0, num_x) / num_x
eta = numpy.linspace(0, num_y) / num_y

d_bottom = numpy.zeros([1, num_x])
d_left = numpy.zeros([1, num_x])
d_top = numpy.zeros([1, num_x])
d_right = numpy.zeros([1, num_x])

# Finding local distances along boundaries
for i in x[1,:]:
    pass