import numpy as np
import cvxpy as cp


"""# Func"""

import random
def Spread(num, size):
  A=[]
  for i in range(num):
    x = random.random()*size
    y = random.random()*size
    A.append([x,y])
  return A

def dist2(a,b):
  dx = a[0]-b[0]
  dy = a[1]-b[1]
  return dx*dx+dy*dy

def get_Bi(SBS,UE,nSBS,nUE,R):
  check = True
  R = 0
  while(check):
    check = False
    B_i = []
    R2 = R*R
    for i in range (nUE):
      SBS_list = []
      for j in range(nSBS):
        if dist2(UE[i],SBS[j]) < R2:
          SBS_list.append(j)
      if len(SBS_list) == 0:
        check = True
        R+=10
        #print('add')
        break
      B_i.append(SBS_list)
  return B_i

def out_p(L,delta):
  p=[]
  
  for i in range(L):
    x = pow((i+1),-delta)
    p.append(x)
  p=np.array(p)
  p = p/p.sum()
  return p

def P2(a, p, Bi, C, S):
  L = len(p)
  nSBS = len(a)
  nUE = len(Bi)
  z = cp.Variable((nUE,L))
  q = cp.Variable((nSBS,L))
  constraint = [0<=q,q<=1,0<=z]
  one = []
  a_cvx = []
  for i in range(nUE):
    a_temp = np.zeros(nSBS)
    for j in Bi[i]:
      #print(j)
      a_temp[j] = a[j]
    a_cvx.append(a_temp)
  for i in range(L):
    one.append(1)
  for i in range(nSBS):
    constraint.append(q[0] @ one <= C/S)

  for i in range(nUE):
    for l in range(L):
      constraint.append(z[i][l]>=1-a_cvx[i] @ q.T[l])

  obj = cp.Minimize((z@p)@np.ones(nUE))
  prob = cp.Problem(obj, constraint)
  result = prob.solve()
  return q.value

from cvxpy.settings import B
import math


def Algo1(q, p, b, Bi,P_a,P_S,duration, alpha, S, arr):

    # init
    K = 30
    nSBS = len(b)
    nUE = len(Bi)
    #print(nUE)
    L = len(p)
    #print(L)
    a = [[random.randint(0, 1) for i in range(nSBS)] for _ in range(K)]
    v = [[random.random() for i in range(nSBS)] for _ in range(K)]
    P = a.copy()
    g = [random.random() for i in range(nSBS)]
    m = [100000000 for i in range(nSBS)]
    M = 100000000
    w = 0.8
    pt = [1 for _ in range(nSBS)]

    for iter in range(num_iter):
        
        for k in range(K):
            temp = 0
            for i in range(nUE):
              for l in range(L):
                temp2 = 0
                for j in Bi[i]:
                  test1 = (a[k][j]*q[j][l])
                  #print("*",test1)
                  temp2 += test1
                #print("**",temp2)
                temp += max(0,1-temp2) * p[l]

            #print(temp)
            e_bh = temp * arr * S * alpha
            F = e_bh
            #print(F)
            for j in range(nSBS):
                pt[j] = a[k][j]*P_a+(1-a[k][j])*P_S
                #print(pt[j]*duration,b[j])
                F += max(0, pt[j]*duration-b[j])
            if F < m[k]:
                m[k] = F
                P[k] = a[k]
            #print(F)
            #return 0
        
        # 9
        for k in range(K):
            M = m[k] if m[k] < M else M
            g = p[k] if m[k] < M else g

        #print(M)
        # 10
        for k in range(K):
            for j in range(nSBS):
                v[k][j] = w*v[k][j] + 2*random.random()*(P[k][j]-a[k][j]) + 2*random.random()*(g[j]-a[k][j])
            for j in range(nSBS):
                a[k][j] = 1 if random.random() <= 1/(1+math.exp(-v[k][j])) else 0
            
            # 19
            for bi in Bi:
                sum = 0
                for j in bi: sum += a[k][j]
                if sum < 1:
                    act = bi[0]
                    for j in bi:
                        act = j if b[j] > b[act] else act
                    a[k][act] = 1


    return M, g

def Algo2(U, B, p, h, num_time_slot, C, S, T, P_0, P_S, P_b, triangle, arr, alpha):
    
    # init
    #Get Bi
    nSBS = len(B)
    nUE = len(U)
    Bi = get_Bi(B,U,nSBS,nUE,R=0) #R can change

    
    # init a
    a = [1 for _ in range(nSBS)]
    #print(a)
    E_old = 0
    E_new = 10000000

    duration = T/num_time_slot
    P_a = triangle * P_b + P_0
    pt = [1 for _ in range(nSBS)]
    # iteration
    while abs(E_old-E_new)/E_new > o:
        # init b
        b = [0 for _ in range(len(B))]
        
        #S4
        #print(len(B))
        
        q = P2(a, p, Bi, C, S)
        #S6 init
        aa = [[1 for _ in range(len(B))] for t in range(num_time_slot)]
        e = [0 for t in range(num_time_slot)]
        #S5 loop
        for t in range(num_time_slot):
            e[t], aa[t] = Algo1(q, p, b, Bi,P_a,P_S,duration, alpha, S, arr) #scalar, vector
            print(t,e[t])
            #S7
            
            for j in range(nSBS):
              pt[j] = aa[t][j]*P_a+(1-aa[t][j])*P_S
              b[j] = max(0, b[j]-pt[j]*duration)+h[t][j] #p[j][t] from (3)
        
        for i in range(nSBS):
            count = 0
            for t in range(num_time_slot):
                count += aa[t][i]
            a[i] = count/t
        print(a)
        E_old = E_new
        #return e
        print("E",sum(e), f'({sum(e)/nSBS})')
        E_new = sum(e)

    return q,aa,E_new

"""# Param"""

Field = 250 #(m)
nSBS = 50
nUE = 40
L = 100
S = 8e7
delta = 0.6
alpha = 5e-8 #J/bits
harvest = 2 #Poisson, pkt/sec
P_0 = 13.6 #W
P_s = 4 #W
P_b = 1 #W
triangle = 4
o = 0.01
C = 40
T = 500 #s
slot = 50
E_p = 5 #J/pkt
arr = 10
num_iter = 50

"""# Main"""

SBS = Spread(nSBS, Field)
UE = Spread(nUE, Field)

# get p
p=out_p(L,delta)

h = np.random.poisson(T/slot*harvest,(nSBS,slot)) * E_p

# sum(sum(h) )/h.size

q,aa,E = Algo2(UE, SBS, p, h, slot, C, S, T, P_0, P_s, P_b, triangle,arr,alpha)
