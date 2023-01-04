import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
x1=['mon','tue','wed','thu','fri']
y1=[300,250,346,589,521]
y2=[751,564,896,321,521]
fig.suptitle('daily sales',fontsize=20)
plt.subplot(1,2,1)
