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
x = [0,0,1,1,0]
s = 17

#declare unknown variables
t = np.zeros((len(I),len(J))).tolist()
d = np.zeros((len(I),len(J))).tolist()

for i in I:
    for j in J:
        t[i][j] = LpVariable("t"+str(i)+str(j), lowBound = 0, upBound = 1, cat = 'Integer')
        d[i][j] = LpVariable("d"+str(i)+str(j), lowBound = 0, upBound = 1, cat = 'Integer')

# defines the problem
prob = LpProblem("problem", LpMaximize)

# defines the objective function to maximize. Must be a formula as a string
objective = ""
for i in I:
    for j in J:
        innerSum = t[i][j]*(alpha*(v[i]+r[i][j])-(1-x[i])*s)
        objective += innerSum

prob += objective

#first constraint
for j in J:
    Tj_sum = 0
    for index,item in enumerate(T[j]):
        Tj_sum += t[item][j]
    prob += (Tj_sum == 0) #add constraint for each j

#second constraint
#t uses indicies not in T[j]
for j in J:
    d_indicies = T[j] #i indicies
    t_indicies = [x for x in I if x not in T[j]]
    d_sum = 0
    t_sum = 0
    for index,item in enumerate(d_indicies):
        d_sum += d[item][j]
    for index,item in enumerate(t_indicies):
        t_sum += t[item][j]
    prob += (t_sum == d_sum) #add constraint for each j



# solve the problem
status = prob.solve()
print("Status:", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Objective function value: ", value(prob.objective))