#############################################################
#
#Orthogonal polynomials expansion for function of 1 variable
#
#
#Version 1.0, 10.08.2021
#
#Author: Andrey Kofnov
#TU Wien
#
#Written for a specific task for https://probing-lab.github.io/ project
#
#This code is a realization of Orthogonal polynomials expansion
#for functions in L2 space with continuous density function
#
#
#

#############################################################
##########                 Example                 ##########
#############################################################
#
# import numpy as np
# import pandas as pd

# from orthogonal_polynomials import Orthogonal_polynomials
# from orthogonal_polynomials import trunc_norm_pdf
#
#
#def mes(x):
#    return trunc_norm_pdf(x, 0, 1, -1, 1)

#params = {}
#params['start'] = -1
#params['stop'] = 1
#params['count'] = 500
#
#
#cl = Orthogonal_polynomials(params, mes, max_poly_degree = 3)
#
#def test_func(x):
#    return np.sin(x) * (np.cos(x) - x**2)
#
#res = cl.Fourier_expansion(test_func, max_poly_degree = 3)
#
#                  p0 = 1           p1 = x         p2 = x**2        p3 = x**3  
####  array([-1.28763867e-16,  9.39245510e-01,  4.92946209e-16, -1.36229668e+00])
#
#
#
##############################################################

import sys
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

from scipy.stats import norm
from tqdm import tqdm


def trunc_norm_pdf(x, mu, sigma, a, b):
    
    return norm.pdf(x, mu, sigma) / (norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma))


def trunc_norm(mu, sigma, a, b):
    
    x = np.random.normal(mu, sigma)
    if x < a:
        return trunc_norm(mu, sigma, a, b)
    if x > b:
        return trunc_norm(mu, sigma, a, b)
    
    return x




class Orthogonal_polynomials:
    
    def __init__(self, params, measure, max_poly_degree = 3):
        
        self.params = params
        self.measure = measure
        self.max_poly_degree = max_poly_degree
        self.orthogonal_poly = None
        self.polys_coef_matrix = None
        self.cur_Fourier_coefficients = None
        
    
    def grid_create(self):
        
        params = self.params

        step = (params['stop'] - params['start']) / params['count']
        tmp = np.arange(params['start'], 
                        params['stop'] + step, 
                        step = step
                        )

        grid = pd.DataFrame({0:tmp})

        return grid

    def integral(self, function):
        
        params = self.params

        _start = params['start']
        _stop = params['stop']
        count = params['count']


        #x = np.linspace(start = _start, stop = _stop, num = count)

        x = self.grid_create()

        y = function(x)

        area = 1
        rect_sqr = 1

        area *= _stop - _start
        rect_sqr = area / count


        try:
            res = y.sum() * rect_sqr
        except:
            res = float(y) * area

        if type(res) == list:
            return (res[0])

        return res
    

    
    def Gram_Smidt_args(self, max_poly_degree = -1):
        
        if max_poly_degree == -1:
            max_poly_degree = self.max_poly_degree

        orthogonal_polys = []

        def poly(x, cur_deg = 0):
            return 1 + x * 0
        orthogonal_polys.append(poly)


        polys_to_orthogonalise = []

        for j in range(1, max_poly_degree + 1):
            def poly(x, cur_deg = j):
                return x**cur_deg
            polys_to_orthogonalise.append(poly)

        return orthogonal_polys, polys_to_orthogonalise


    def Gram_Smidt_process(self, max_poly_degree = -1):
        
        if max_poly_degree == -1:
            max_poly_degree = self.max_poly_degree
        
        orthogonal_polys, polys_to_orthogonalise = self.Gram_Smidt_args(max_poly_degree)
        
        measure = self.measure 
        params = self.params 

        orthogonal_poly = orthogonal_polys.copy()

        #################################################################
        ### Create a matrix with coefficients for initial polynomials ###
        #################################################################

        len_ort_polys = len(orthogonal_polys)

        polys_coef_matrix = np.zeros([len(polys_to_orthogonalise) + len_ort_polys, 
                                      len(polys_to_orthogonalise) + len_ort_polys
                                     ]
                                    )

        np.fill_diagonal(polys_coef_matrix, 1)

        #################################################################

        i = 0

        for pol_num, new_poly in enumerate(polys_to_orthogonalise):

            #tmp_poly = new_poly

            def tmp_poly(x, cur_new = new_poly):
                return cur_new(x)

            # print(i)

            for it, ort_poly in enumerate(orthogonal_poly):


                # func_inner_prod = lambda x: ort_poly(x) * new_poly(x) * measure(x)
                def func_inner_prod(x, cur_ort = ort_poly, cur_new = new_poly):
                    return cur_ort(x) * cur_new(x) * measure(x)

                # func_sq_norm = lambda x: ort_poly(x) * ort_poly(x) * measure(x)
                def func_sq_norm(x, cur_ort = ort_poly):
                    return cur_ort(x) ** 2 * measure(x)

                #######################################################
                ##### Here is an update of coefficients in matrix #####
                #######################################################

                coef1 = self.integral(func_inner_prod) / self.integral(func_sq_norm)
     
                for i in range(pol_num+len_ort_polys):
                    polys_coef_matrix[pol_num+len_ort_polys, i] -= coef1 * polys_coef_matrix[it, i]

                #######################################################

                #tmp_poly = lambda x, curr_poly = tmp_poly: curr_poly(x) - (integral(func_inner_prod, _start, _stop, count) / integral(func_sq_norm, _start, _stop, count)) * ort_poly(x)
                def tmp_poly(x, cur_ort = ort_poly, curr_poly = tmp_poly, inner_prod = func_inner_prod, sq_norm = func_sq_norm):

                    coef = self.integral(inner_prod) / self.integral(sq_norm)

                    return curr_poly(x) - coef * cur_ort(x)


            #func_poly_norm = lambda x: tmp_poly(x) * tmp_poly(x) * measure(x)
            def func_poly_norm(x, curr_poly = tmp_poly):
                return curr_poly(x) ** 2 * measure(x)

            tmp_poly_norm = np.sqrt(self.integral(func_poly_norm))

            ######################################################################
            ####### Division of coefficients in matrix by the norm-value #########
            ######################################################################

            for i in range(pol_num + len_ort_polys + 1):
                    polys_coef_matrix[pol_num+len_ort_polys, i] /= tmp_poly_norm

            ######################################################################

            def res_poly(x, curr_poly = tmp_poly, cur_poly_norm = tmp_poly_norm):

                return curr_poly(x) / cur_poly_norm

            orthogonal_poly.append(res_poly)

            i += 1
            
        self.orthogonal_poly = orthogonal_poly
        self.polys_coef_matrix = polys_coef_matrix

        return orthogonal_poly, polys_coef_matrix


    def Fourier_expansion_ort(self, function):
        
        if type(self.orthogonal_poly) == type(None):
            print('Orthogonal polynomials are not estimated!')
            return []
        
        polys = self.orthogonal_poly
        measure = self.measure
        params = self.params
        
        coef = []

        for poly in tqdm(polys):

            def inner_product(x, cur_pol = poly): 
                return cur_pol(x) * function(x) * measure(x)

            coef.append(self.integral(inner_product))
            
        self.cur_Fourier_coefficients = coef

        return coef


    def Fourier_expansion(self, function, max_poly_degree = -1):
        
        if max_poly_degree == -1:
            max_poly_degree = self.max_poly_degree
        
        measure = self.measure
        params = self.params

        orthogonal_poly, polys_coef_matrix = self.Gram_Smidt_process(max_poly_degree)

        coef = self.Fourier_expansion_ort(function)
        
        self.cur_Fourier_coefficients = coef

        Result = np.zeros(max_poly_degree + 1)

        for i in range(polys_coef_matrix.shape[0]):
            polys_coef_matrix[i] = [x * coef[i] for x in polys_coef_matrix[i]]

        for i in range(polys_coef_matrix.shape[1]):
            Result[i] = sum(polys_coef_matrix[:, i])
            
        self.cur_mono_coefficients = Result

        return Result


    def function_estimator(self, x, res_coef = None):
        
        if type(res_coef) == type(None):
            res_coef = self.cur_mono_coefficients
        
        if type(self.cur_mono_coefficients) == type(None):
            print('Coefficients are not estimated.')
            return

        res = 0
        for j in range(res_coef.shape[0]):
            res += res_coef[j] * (x**j)

        return res

    def error_estimator(self, function, max_poly_degree = -1, plot = False):
        
        if max_poly_degree == -1:
            max_poly_degree = self.max_poly_degree
        
        measure = self.measure
        params = self.params 

        coef_dict = {}

        result = np.zeros(max_poly_degree + 1)

        for j in range(1, max_poly_degree + 1):
            coef_dict[j] = self.Fourier_expansion(function, max_poly_degree = j)

        for j in range(1, max_poly_degree + 1):

            def error_est(x, pol_num = j):
                return function(x) - self.function_estimator(x, coef_dict[j])

            def norm_sq(x, pol_num = j):
                return (error_est(x, pol_num = j) ** 2 * measure(x))

            result[j - 1] = np.sqrt(self.integral(norm_sq))
           
        
        if plot == True: 
            y_dict = {}
            x = self.grid_create()
            y = function(x)
            plt.plot(x, y, label = 'function')
            for j in range(1, max_poly_degree + 1):
                y_dict[j] = self.function_estimator(x, res_coef = coef_dict[j])
                plt.plot(x, y_dict[j], label = 'degree ' + str(j))
            plt.legend()
        return result