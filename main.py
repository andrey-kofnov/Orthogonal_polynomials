#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#--------------------- Examples calculation ---------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

from ort_poly2 import *
import numpy as np
import math
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import nquad
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.warn("deprecated", DeprecationWarning)
#----------------------------------------------------------------------------------
example = 1

mu_1 = 0
sigma_1 = 1
a_1 = -math.inf
b_1 = math.inf

mu_2 = 2
sigma_2 = 0.1
a_2 = -math.inf
b_2 = math.inf

specification = '''
func1(x1, x2) = ksi*mp.exp(-x1) + (ksi - 0.5*ksi*ksi) * mp.exp(x2-x1), ksi = 0.3
x1 ~ Normal({0}, {1})
x2 ~ Normal({2}, {3})
'''.format(mu_1, sigma_1**2, mu_2, sigma_2**2)

with open("Errors, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')
with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')


def mes_11(x):
    return norm_pdf(x, mu_1, sigma_1, infty = True)

def mes_12(x):
    return norm_pdf(x, mu_2, sigma_2, infty = True)


mes1 = {}
mes1[1] = mes_11
mes1[2] = mes_12

params1 = {}
params1[1] = {}
params1[2] = {}

params1[1]['start'] = a_1
params1[1]['stop'] = b_1
params1[1]['opts'] = {'epsabs' : 1.49e-7}

params1[2]['start'] = a_2
params1[2]['stop'] = b_2
params1[2]['opts'] = {'epsabs' : 1.49e-7}

def func1(x1, x2):
    ksi = 0.3
    return ksi*mp.exp(-x1) + (ksi - 0.5*ksi*ksi) * mp.exp(x2-x1)

for j in [1, 2, 3, 4, 5]:
    X = K(params1, mes1, max_poly_degree=j, infty=True)
    Pol = X.Fourier_expansion(func1)
    with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(Pol)+'\n')
        handle.write('-'*15 + '\n')
    print(Pol)
    er = X.error_estimator(func1)
    with open("Errors, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(er)+'\n')
        handle.write('-' * 15 + '\n')
    print('Degree:', j, " , error: ", er)

#----------------------------------------------------------------------------------
example = 2

mu_1 = 4
sigma_1 = 1
a_1 = 3
b_1 = 5

mu_2 = 2
sigma_2 = 0.1
a_2 = 0
b_2 = 4

specification = '''
func2(x1, x2) = 0.3*np.exp(x1-x2) + 0.6*np.exp(-x2)
x1 ~ TruncNormal({0}, {1}, [{2}, {3}])
x2 ~ TruncNormal({4}, {5}, [{6}, {7}])
'''.format(mu_1, sigma_1**2, a_1, b_1, mu_2, sigma_2**2, a_2, b_2)

with open("Errors, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')
with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')


def mes_21(x):
    return trunc_norm_pdf(x, mu_1, sigma_1, a_1, b_1)

def mes_22(x):
    return trunc_norm_pdf(x, mu_2, sigma_2, a_2, b_2)

mes2 = {}
mes2[1] = mes_21
mes2[2] = mes_22

params2 = {}
params2[1] = {}
params2[2] = {}

params2[1]['start'] = a_1
params2[1]['stop'] = b_1
params2[1]['opts'] = {'epsabs' : 1.49e-7}

params2[2]['start'] = a_2
params2[2]['stop'] = b_2
params2[2]['opts'] = {'epsabs' : 1.49e-7}

print("Test measure 1: ",
      nquad(mes_21,
            [[params2[1]['start'],
                      params2[1]['stop']]
                     ],
            opts=params2[1]['opts'],
            full_output=False)[0]
      )

print("Test measure 2: ",
      nquad(mes_22,
            [[params2[2]['start'],
              params2[2]['stop']]
             ],
            opts=params2[2]['opts'],
            full_output=False)[0]
      )

def func2(x1, x2):
    return 0.3*np.exp(x1-x2) + 0.6*np.exp(-x2)

x_1 = list(np.linspace(a_1, b_1, 100)) * 100
x_2 = sorted(list(np.linspace(a_2, b_2, 100)) * 100)
y1 = [func2(i, j) for i, j in zip(x_1, x_2)]
y_est = {}

for j in [1, 2, 3, 4, 5]:
    X = K(params2, mes2, max_poly_degree=j, infty=False)
    Pol = X.Fourier_expansion(func2)
    with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(Pol)+'\n')
        handle.write('-'*15 + '\n')
    y_est[j] = [float(Pol.subs('x1', i).subs('x2', j)) for i, j in zip(x_1, x_2)]
    print(Pol)
    er = X.error_estimator(func2)
    with open("Errors, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(er)+'\n')
        handle.write('-' * 15 + '\n')
    print('Degree:', j, " , error: ", er)

F2 = plt.figure(figsize=(10, 8))
ax = F2.gca(projection='3d')
ax.plot_trisurf(x_1, x_2, y1, label = 'Function')
leg = ['Function']

for j in y_est.keys():
    ax.plot_trisurf(x_1, x_2, y_est[j], label = 'Estimator ' + str(j))
    leg.append('Estimator ' + str(j))

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
fakeline = []
for j in ['b', 'g', 'r', 'c', 'y', 'm'][0:len(leg)]:
    fakeline.append(mpl.lines.Line2D([0],[0], linestyle="none", c=j, marker = 'o'))
ax.legend(fakeline, leg)

F2.savefig('plot{0}.pdf'.format(example))

#----------------------------------------------------------------------------------
example = 3

mu_1 = 4
sigma_1 = 1
a_1 = 3
b_1 = 5

shape_2 = 1
scale_2 = 3
a_2 = 0.5
b_2 = 1

specification = '''
func3(x1, x2) = np.exp(x1*x2)
x1 ~ TruncNormal({0}, {1}, [{2}, {3}])
x2 ~ TruncGamma({4}, {5}, [{6}, {7}])
'''.format(mu_1, sigma_1**2, a_1, b_1, scale_2, shape_2, a_2, b_2)

with open("Errors, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')
with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')


def mes_31(x):
    return trunc_norm_pdf(x, mu_1, sigma_1, a_1, b_1)

def mes_32(x):
    return gamma.pdf(x, a = shape_2, scale = scale_2) / \
           (gamma.cdf(b_2, a = shape_2, scale = scale_2) - gamma.cdf(a_2, a = shape_2, scale = scale_2))

mes3 = {}
mes3[1] = mes_31
mes3[2] = mes_32

params3 = {}
params3[1] = {}
params3[2] = {}

params3[1]['start'] = a_1
params3[1]['stop'] = b_1
params3[1]['opts'] = {'epsabs' : 1.49e-7}

params3[2]['start'] = a_2
params3[2]['stop'] = b_2
params3[2]['opts'] = {'epsabs' : 1.49e-7}

print("Test measure 1: ",
      nquad(mes_31,
            [[params3[1]['start'],
                      params3[1]['stop']]
                     ],
            opts=params3[1]['opts'],
            full_output=False)[0]
      )

print("Test measure 2: ",
      nquad(mes_32,
            [[params3[2]['start'],
              params3[2]['stop']]
             ],
            opts=params3[2]['opts'],
            full_output=False)[0]
      )

def func3(x1, x2):
    return np.exp(x1*x2)

x_1 = list(np.linspace(a_1, b_1, 100)) * 100
x_2 = sorted(list(np.linspace(a_2, b_2, 100)) * 100)
y1 = [func3(i, j) for i, j in zip(x_1, x_2)]
y_est = {}

for j in [1, 2, 3, 4, 5]:
    X = K(params3, mes3, max_poly_degree=j, infty=False)
    Pol = X.Fourier_expansion(func3)
    with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(Pol)+'\n')
        handle.write('-'*15 + '\n')
    y_est[j] = [float(Pol.subs('x1', i).subs('x2', j)) for i, j in zip(x_1, x_2)]
    print(Pol)
    er = X.error_estimator(func3)
    with open("Errors, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(er)+'\n')
        handle.write('-' * 15 + '\n')
    print('Degree:', j, " , error: ", er)

F3 = plt.figure(figsize=(10, 8))
ax = F3.gca(projection='3d')
ax.plot_trisurf(x_1, x_2, y1, label = 'Function')
leg = ['Function']

for j in y_est.keys():
    ax.plot_trisurf(x_1, x_2, y_est[j], label = 'Estimator ' + str(j))
    leg.append('Estimator ' + str(j))

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
fakeline = []
for j in ['b', 'g', 'r', 'c', 'y', 'm'][0:len(leg)]:
    fakeline.append(mpl.lines.Line2D([0],[0], linestyle="none", c=j, marker = 'o'))
ax.legend(fakeline, leg)

F3.savefig('plot{0}.pdf'.format(example))

#----------------------------------------------------------------------------------
example = 4

mu_1 = 4
sigma_1 = 1
a_1 = 3
b_1 = 5

shape_2 = 1
scale_2 = 3
a_2 = 0.5
b_2 = 1

a_3 = 4
b_3 = 8

specification = '''
func4(x1, x2, x3) = 0.3*np.exp(x1-x2) + 0.6*np.exp(x2-x3) + 0.1*np.exp(x3-x1)
x1 ~ TruncNormal({0}, {1}, [{2}, {3}])
x2 ~ TruncGamma({4}, {5}, [{6}, {7}])
x3 ~ Uniform({8}, {9})
'''.format(mu_1, sigma_1**2, a_1, b_1, scale_2, shape_2, a_2, b_2, a_3, b_3)

with open("Errors, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')
with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')


def mes_41(x):
    return trunc_norm_pdf(x, mu_1, sigma_1, a_1, b_1)

def mes_42(x):
    return gamma.pdf(x, a = shape_2, scale = scale_2) / \
           (gamma.cdf(b_2, a = shape_2, scale = scale_2) - gamma.cdf(a_2, a = shape_2, scale = scale_2))

def mes_43(x):
    return 1 / (b_3 - a_3)


mes4 = {}
mes4[1] = mes_41
mes4[2] = mes_42
mes4[3] = mes_43

params4 = {}
params4[1] = {}
params4[2] = {}
params4[3] = {}

params4[1]['start'] = a_1
params4[1]['stop'] = b_1
params4[1]['opts'] = {'epsabs' : 1.49e-7}

params4[2]['start'] = a_2
params4[2]['stop'] = b_2
params4[2]['opts'] = {'epsabs' : 1.49e-7}

params4[3]['start'] = a_3
params4[3]['stop'] = b_3
params4[3]['opts'] = {'epsabs' : 1.49e-7}


print("Test measure 1: ",
      nquad(mes_41,
            [[params4[1]['start'],
                      params4[1]['stop']]
                     ],
            opts=params4[1]['opts'],
            full_output=False)[0]
      )

print("Test measure 2: ",
      nquad(mes_42,
            [[params4[2]['start'],
              params4[2]['stop']]
             ],
            opts=params4[2]['opts'],
            full_output=False)[0]
      )


print("Test measure 3: ",
      nquad(mes_43,
            [[params4[3]['start'],
                      params4[3]['stop']]
                     ],
            opts=params4[3]['opts'],
            full_output=False)[0]
      )

def func4(x1, x2, x3):
    return 0.3*np.exp(x1-x2) + 0.6*np.exp(x2-x3) + 0.1*np.exp(x3-x1)


for j in [1, 2, 3]:
    X = K(params4, mes4, max_poly_degree=j, infty=False)
    Pol = X.Fourier_expansion(func4)
    with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(Pol)+'\n')
        handle.write('-'*15 + '\n')
    print(Pol)
    er = X.error_estimator(func4)
    with open("Errors, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(er)+'\n')
        handle.write('-' * 15 + '\n')
    print('Degree:', j, " , error: ", er)


#----------------------------------------------------------------------------------

example = 5

mu_1 = 0
sigma_1 = 1
a_1 = -math.inf
b_1 = math.inf

specification = '''
func5(x1) = 0.3*mp.cos(x1) + 0.7*mp.sin(x1)
x1 ~ Normal({0}, {1})
'''.format(mu_1, sigma_1**2)

with open("Errors, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')
with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')


def mes_51(x):
    return norm_pdf(x, mu_1, sigma_1, infty = True)


mes5 = {}
mes5[1] = mes_51

params5 = {}
params5[1] = {}

params5[1]['start'] = a_1
params5[1]['stop'] = b_1
params5[1]['opts'] = {'epsabs' : 1.49e-7}

def func5(x1):
    return 0.3*mp.cos(x1) + 0.7*mp.sin(x1)

x_1 = list(np.linspace(a_1, b_1, 100))

y1 = [func5(i) for i in x_1]
y_est = {}

for j in [1, 2, 3, 4, 5]:
    X = K(params5, mes5, max_poly_degree=j, infty=True)
    Pol = X.Fourier_expansion(func5)
    with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(Pol)+'\n')
        handle.write('-'*15 + '\n')
    #y_est[j] = [float(Pol.subs('x1', i) for i in x_1)]
    print(Pol)
    er = X.error_estimator(func5)
    with open("Errors, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(er)+'\n')
        handle.write('-' * 15 + '\n')
    print('Degree:', j, " , error: ", er)


#----------------------------------------------------------------------------------

example = 6

mu_1 = 2
sigma_1 = 0.1
a_1 = 1
b_1 = 3

a_2 = 1
b_2 = 2

specification = '''
func6(x1, x2) = np.log(x1 + x2)
x1 ~ TruncNormal({0}, {1}, [{2}, {3}])
x2 ~ Uniform({4}, {5})
'''.format(mu_1, sigma_1**2, a_1, b_1, a_2, b_2)

with open("Errors, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')
with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
    handle.write(specification + '\n')


def mes_61(x):
    return trunc_norm_pdf(x, mu_1, sigma_1, a_1, b_1)

def mes_62(x):
    return 1 / (b_2 - a_2)


mes6 = {}
mes6[1] = mes_61
mes6[2] = mes_62

params6 = {}
params6[1] = {}
params6[2] = {}

params6[1]['start'] = a_1
params6[1]['stop'] = b_1
params6[1]['opts'] = {'epsabs' : 1.49e-7}

params6[2]['start'] = a_2
params6[2]['stop'] = b_2
params6[2]['opts'] = {'epsabs' : 1.49e-7}


print("Test measure 1: ",
      nquad(mes_61,
            [[params6[1]['start'],
              params6[1]['stop']]
             ],
            opts=params6[1]['opts'],
            full_output=False)[0]
      )

print("Test measure 2: ",
      nquad(mes_62,
            [[params6[2]['start'],
              params6[2]['stop']]
             ],
            opts=params6[2]['opts'],
            full_output=False)[0]
      )


def func6(x1, x2):
    return np.log(x1 + x2)


for j in [1, 2]:
    X = K(params6, mes6, max_poly_degree=j, infty=False)
    Pol = X.Fourier_expansion(func6)
    with open("Polynomials, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(Pol)+'\n')
        handle.write('-'*15 + '\n')
    print(Pol)
    er = X.error_estimator(func6)
    with open("Errors, example_{0}.txt".format(example), 'a') as handle:
        handle.write(str(j)+': ' + str(er)+'\n')
        handle.write('-' * 15 + '\n')
    print('Degree:', j, " , error: ", er)