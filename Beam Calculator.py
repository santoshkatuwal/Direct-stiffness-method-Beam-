#Beam Solver using stiffness matrix method
#Unyielding support condition
#Unit=kN,m
#Where, K=Pj-Pl(PL=member reaction due to given load, Pj=given joint load)

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PyQt5 import QtWidgets
 
#%% Data Input


path=os.getcwd()
data=pd.read_csv(path+'\\input.csv')
NODES=data.Nodes.dropna()
X=data.x.dropna()
Y=data.y.dropna()
P_joint=data.P_joint.dropna()
M_joint=data.M_joint.dropna()
L_all=data.L.dropna()
EI_all=data.EI.dropna()

#formatting data input
N_i=data.node_i.dropna()
N_j=data.node_j.dropna()
dy=data.defln_y.dropna()
rot=data.rot.dropna()
M_i=data.FEM_frd.dropna()
M_j=data.FEM_back.dropna()
R_i=data.R_i.dropna()
R_j=data.R_j.dropna()


#%%Formulation of global stiffness matrix
l=len(NODES)
elem=len(N_i)
dof_total=4*l
K_elem=np.zeros([4,4])
n=elem*2+2
K_global=np.zeros([n,n])
K_all=[]
for i in range(0,elem):
    K=[]
    EI=EI_all[i]
    L=L_all[i]
    K.append(12*EI/L**3)
    K.append(6*EI/L**2)
    K.append(-12*EI/L**3)
    K.append(6*EI/L**2)
    K.append(6*EI/L**2)
    K.append(4*EI/L)
    K.append(-6*EI/L**2)
    K.append(2*EI/L)
    K.append(-12*EI/L**3)
    K.append(-6*EI/L**2)
    K.append(12*EI/L**3)
    K.append(-6*EI/L**2)
    K.append(6*EI/L**2)
    K.append(2*EI/L)
    K.append(-6*EI/L**2)
    K.append(4*EI/L)

    #making element stiffness matrix of size equal to dummy global matrix by padding
    K=np.array(K)
    K=K.reshape(4,4)
    #Printing inividual stiffness matrix
    print("\n<<<<<Stiffness Matrix K"+str(i+1)+">>>>>\n")
    print(K)
    K=np.pad(K,(i*2,(n-4)-i*2))
    #storing matrix value in dynamic variable: K1, K2, K3......
    globals()[f"K{i+1}"] = K
    #Creating Global Matrix
    K_global=K_global+K

#%%Connectivity check of members
#Computing total FEM and member reactions at each ends
P=[]
for i in NODES:
    FEM_total=0
    R_total=0
    a_i=N_i.index[N_i==i].tolist()
    a_j=N_j.index[N_j==i].tolist()
    for j in a_i:
        FEM_total=FEM_total+M_i[j]
        R_total=R_total+R_i[j]
    for k in a_j:
        FEM_total=FEM_total+M_j[k]
        R_total=R_total+R_j[k]
    P.append(R_total)
    P.append(-1*FEM_total)
    
    #Individual nodal loads eg. FEM_A, FEM_B,....
    globals()[f"FEM_{i}"]=FEM_total
    globals()[f"R_{i}"]=R_total
P=np.array(P)

#%%Formatting given joint load matrix
P_JOINT=[] #(Given joint load)
for i in range(0,len(NODES)):
    P_JOINT.append(float(P_joint[i]))
    P_JOINT.append(float(-1*M_joint[i]))

P_JOINT=np.array(P_JOINT)
#Combined load of given joint load and member reaction (PL_Pj)
P=P_JOINT-P #P=matrix due to given loading condition
            #P_JOINT= given joint load
        
#%%Droping rows and columns having 0 deflection and rotation
#Merging and ordering
delta=[]
for i in range(0,len(NODES)):
    delta.append(float(dy[i]))
    delta.append((rot[i]))
delta=pd.Series(delta)
#identifying row index having no deflection and rotation
for i in range(0,n):
    drop_me=delta.index[delta!="x"].tolist()

#Dropping cells
k_global=np.delete(K_global,drop_me,axis=0)
k_global=np.delete(k_global,drop_me,axis=1)
rxn=np.delete(P,drop_me)

#%%Calculating Unknown deflections and rotations
inv_k=np.linalg.inv(k_global) #inverse of global stiffness matrix
defln=np.matmul(inv_k,rxn)

#Overwriting unknown deflection 'x' with computed values
for i in range(0,n):
    ind_unknown=delta.index[delta=="x"].tolist()
for i in range(0,len(ind_unknown)):
    delta[ind_unknown[i]]=defln[i]
delta=np.array(delta)
delta=delta.astype(float)

#Final reaction calculation
R=np.matmul(K_global,delta)-P
R=np.round(R,3)

Result=pd.DataFrame({"Deflection":delta,
                     "Reaction":R})
print("\n<<<<<<Global Stiffness Matrix>>>>>>\n")
print(K_global)

print("\n<<<<<<Reaction Forces>>>>>>\n")
print(Result)
