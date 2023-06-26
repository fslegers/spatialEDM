#https://pysindy.readthedocs.io/en/latest/examples/15_pysindy_lectures/example.html

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0,100,1)
t = t/100.0

p = [1,1]
q = [2,1]
x = [1,3]
y = [2,2]
r_x = x[0] + t * (y[0] - x[0])
r_y = x[1] + t * (y[1] - x[1])
s_x = p[0] + t * (q[0] - p[0])
s_y = p[1] + t * (q[1] - p[1])

plt.plot(r_x, r_y, label = "r(t)", color = 'blue')
plt.plot(s_x, s_y, label = "s(t)", color = 'orange')
plt.plot(p[0], p[1], marker = 'o', color = 'black')
plt.plot(q[0], q[1], marker = 'o', color = 'black')
plt.plot(x[0], x[1], marker = 'o', color = 'black')
plt.plot(y[0], y[1], marker = 'o', color = 'black')
plt.text(1.4, 2.7, "s(t)", fontdict=None, color = 'blue', fontsize = 13)
plt.text(1.4, 1.1, "r(t)", fontdict=None, color = 'orange', fontsize = 13)
plt.text(1,1.15,"p")
plt.text(2,1.15,"q")
plt.text(1,3.15,"x")
plt.text(2,2.15,"y")
plt.ylim((0,5))
plt.show()

p = [1,1]
q = [2,4]
x = [1,3]
y = [2,2]
r_x = x[0] + t * (y[0] - x[0])
r_y = x[1] + t * (y[1] - x[1])
s_x = p[0] + t * (q[0] - p[0])
s_y = p[1] + t * (q[1] - p[1])

plt.plot(r_x, r_y, color = 'blue')
plt.plot(s_x, s_y, color = 'orange')
plt.plot(p[0], p[1], marker = 'o', color = 'black')
plt.plot(q[0], q[1], marker = 'o', color = 'black')
plt.plot(x[0], x[1], marker = 'o', color = 'black')
plt.plot(y[0], y[1], marker = 'o', color = 'black')
plt.text(1.4, 2.7, "s(t)", fontdict=None, color = 'blue', fontsize = 13)
plt.text(1.4, 2.0, "r(t)", fontdict=None, color = 'orange', fontsize = 13)
plt.text(1,1.15,"p")
plt.text(2,4.15,"q")
plt.text(1,3.15,"x")
plt.text(2,2.15,"y")
plt.ylim((0,5))
plt.show()




