import matplotlib.pyplot as plt

f = open("input.txt", "r");

L = []

for s in f:
    row = (s.split())
    row = [float (i) for i in row]
    L.append(row)
 #   print(L)
f.close()

x = [row[0] for row in L]
y = [row[4] for row in L]

axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([0.89,0.91])

plt.plot(x , y)
plt.ylabel('Accuracy')
plt.xlabel('Learning rate')
plt.title('Single Layer Neural Net  (epochs = 5000, # hidden layer inputs = 128, batch size = 500)')
plt.show()