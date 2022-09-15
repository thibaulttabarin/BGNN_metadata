from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure()
unif = [float(i.strip()) for i in open('../out_true.txt').readlines()]
not_unif = [float(i.strip()) for i in open('../out_false.txt').readlines()]
plt.hlines(1, 1, 90)

subfigs = fig.subplots(2, 1)

subfigs[0].hlines(1, 1, 90)
subfigs[0].eventplot(unif, orientation='horizontal', colors='b')
subfigs[0].axes.get_yaxis().set_visible(False)

subfigs[1].hlines(1, 1, 90)
subfigs[1].eventplot(not_unif, orientation='horizontal', colors='r')
subfigs[1].axes.get_yaxis().set_visible(False)

print(sum(unif)/len(unif))
print(sum(not_unif)/len(not_unif))

plt.show()
