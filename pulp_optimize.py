from pulp import *
import numpy as np

#initialize known parameters
I = [0,1,2,3,4]
J = [0,1,2]
T = np.array([ #T is a matrix of hospitals associated to each patient
    [1,2,4], #patient 0
    [0,1,2], #patient 1
    [0,2,4]  #patient 2
]) 
alpha = 7
v = [4,2,7,9,1] #of length I
r = np.array([ #IxJ matrix
    [2,3,4],
    [7,6,8],
    [1,1,2],
    [7,7,7],
    [9,0,4]
])
e = [0,0,1,1,0]
s = 17

#declare unknown variables
t = np.zeros((len(I),len(J))).tolist()
y = np.zeros((len(I),len(J))).tolist()

index = 0
for i in I:
    for j in J:
        t[i][j] = LpVariable("t"+str(index), lowBound = 0, upBound = 1, cat = 'Integer')
        y[i][j] = LpVariable("y"+str(index), lowBound = 0, upBound = 1, cat = 'Integer')
        index += 1

# defines the problem
prob = LpProblem("problem", LpMaximize)

# defines the objective function to maximize. Must be a formula as a string
objective = ""
for i in I:
    for j in J:
        innerSum = t[i][j]*(alpha*(v[i]+r[i][j])-(1-e[i])*s)
        objective += innerSum

prob += objective

#first constraint
for j in J:
    Tj_sum = 0
    for index,item in enumerate(T[j]):
        Tj_sum += t[index][j]
    prob += (Tj_sum == 0) #add constraint 1 for each j

#second constraint
"""for i in I:
    for j in J:
        if(t[i][j] == 1):
            prob += (y[i][j] != 1)"""

# solve the problem
status = prob.solve()
print("Status:", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Objective function value: ", value(prob.objective))
