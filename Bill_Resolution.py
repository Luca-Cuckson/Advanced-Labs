import numpy
import matplotlib.pyplot as plt

def efficiency_graph(r,d,A_0,decay_const,t,A_out,E):
    A_t=numpy.zeros(len(A_0))
    A_in=numpy.zeros(len(A_0))
    efficiency=numpy.zeros(len(A_0))
    for i in range(0,len(A_0)):
        A_t[i]=A_0[i]*numpy.exp(-decay_const[i]*t[i])
        A_in[i]=r**2/(4*d**2)*A_t[i]
        efficiency[i]=A_out[i]/A_in[i]*100
    print(A_in)
    print(efficiency)
    plt.figure(1)
    plt.scatter(E,efficiency)
    plt.savefig("efficiency.png", bbox_inches="tight")

def resolution_graph(delta_E,E_0):
    resolution=numpy.zeros(len(delta_E))
    for i in range(0,len(delta_E)):
        resolution[i]=delta_E[i]/E_0[i]*100
    print(resolution)
    plt.figure(2)
    plt.plot(E_0,resolution)
    plt.savefig("resolution.png",bbox_inches="tight")

delta_E=[567.17-497.87,747.87-666.32,1350.76-1223.36,138.60] #FWHM of the energy peak
E_0=[534.65,709.49,1291.05,1399.08] #Energy at centre of peak
resolution_graph(delta_E,E_0)

r=59.2/2
d=83+3.956+9.28-27.85
A_0=[379369.2] #Activity of source from source table in Bq (Not sure if current value is correct for Na22 source)
decay_const=[0.267] #In years^-1
t=[9.67] #Time since purchase in years
counts=[2810000]#,3208880,534003] #Number of counts over whole peak
running_time=[5589.04]#,4651.94,5589.04,244860.88] #In seconds
A_out=[0]#,0,0,0]
for i in range (0,len(counts)):
    A_out[i]=counts[i]/running_time[i]
E=[534.65]#,709.49,1291.05,1399.08] #Energy at centre of peak
efficiency_graph(r,d,A_0,decay_const,t,A_out,E)