import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [i**2 for i in x]
z = [i**3 for i in x]
fig, axes = plt.subplots(1,3,figsize=(10,3.3))
axes[0].plot(x,y,color='r',marker='*',label ='2-square')
axes[0].plot(x,z,color='b',marker='o',label ='3-square')
axes[0].set_xlabel("fig1")
axes[0].set_ylabel("value")
axes[0].legend(loc='upper right')
axes[1].plot(x,y)
axes[2].plot(x,y)

plt.show()