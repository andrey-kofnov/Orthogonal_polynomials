import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mpmath as mp
mp.dps = 20

import scipy.integrate as si
import scipy
from scipy.stats import norm
import numba
from numba import cfunc,carray
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from scipy.integrate import nquad

from sympy import *
from sympy import hermite_poly, sin, cos, exp, diff, pi, sqrt, erf
from sympy import re, im

import random
from random import choice as rand_choice
from random import random
import math
from tqdm import tqdm

from itertools import product as itert_prod

print('Current recursion limit: ', sys.getrecursionlimit())

    
    
def norm_pdf(x, mu, sigma, infty = False):
    if infty == True:
        return float((1 / math.sqrt(2 * math.pi) / sigma) * math.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    return float((1 / np.sqrt(2 * np.pi) / sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2)))

def norm_cdf(x, mu, sigma, infty = False):
    
    if infty == True:
        #def pdf(y):
        #    return norm_pdf(y, mu, sigma, infty)
        
        #return float(mp.quad(pdf, [-math.inf, x]))
        return float(mp.ncdf(x, mu, sigma))
    
    return norm.cdf(x, mu, sigma)
    
def trunc_norm_pdf(x, mu, sigma, a, b, infty = False):

    if (x < a) or (x > b):
        return 0.0
    
    if infty == True:
        return norm_pdf(x, mu, sigma, True) / float(norm_cdf(b, mu, sigma, True) - norm_cdf(a, mu, sigma, True))
    
    return norm_pdf(x, mu, sigma) / float(norm_cdf(b, mu, sigma) - norm_cdf(a, mu, sigma))
	
	



class K:
    
    def __init__(self, params, measure, max_poly_degree = 3, infty = False):
        
        self.params = params
        self.measure = measure
        self.max_poly_degree = max_poly_degree
        self.is_Infty = infty
        self.orthogonal_poly = None
        self.polys_coef_matrix = None
        self.orthogonal_poly_collection = None
        self.polys_coef_matrix_collection = None
        self.poly_combs = None
        self.cur_Fourier_coefficients = None
        self.poly_dict_frame = None
        self._vars = None
        
        
    def integral(self, 
                 func, 
                 params, 
                 infty = False
                ):
        
        if infty == True:
            return float(mp.quadgl(lambda x: func(x), 
                                 [params['start'], params['stop']]
                                )
                        )
            
        return nquad(func, 
                     [[params['start'], params['stop']]], 
                     opts = params['opts'], 
                     full_output=False)[0]
    
    def Fourier_integral(self, 
                         func, 
                         params, 
                         infty = False
                        ):
        
        _range = []
        opts0 = []
        for j in params.keys():
            tmp_r = [params[j]['start'], params[j]['stop']]
            _range.append(tmp_r)
            
            tmp_op = params[j]['opts']
            opts0.append(tmp_op)
            
        if infty == True:
            
            if len(params.keys()) == 1:
                return float(mp.quad(lambda x: func(x), 
                                     [params[1]['start'], params[1]['stop']]
                                    )
                            )
            
            elif len(params.keys()) == 2:
                return float(mp.quad(lambda x, y: func(x, y), 
                                     [params[1]['start'], params[1]['stop']], 
                                     [params[2]['start'], params[2]['stop']], 
                                     method='tanh-sinh'
                                    )
                            )
            elif len(params.keys()) == 3:
                return float(mp.quad(lambda x, y, z: func(x, y, z), 
                                     [params[1]['start'], params[1]['stop']], 
                                     [params[2]['start'], params[2]['stop']], 
                                     [params[3]['start'], params[3]['stop']]
                                    )
                            )
        
            raise ValueError('Too many variables. With infinity maximum 3 arguments.')
        
        return nquad(func, _range, opts = opts0, full_output=False)[0]
    
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


    def Gram_Smidt_process(self, var_index = 1, max_poly_degree = -1):
        
        if max_poly_degree == -1:
            max_poly_degree = self.max_poly_degree
        
        orthogonal_polys, polys_to_orthogonalise = self.Gram_Smidt_args(max_poly_degree)
        
        measure = self.measure[var_index] 
        params = self.params[var_index] 

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

            #print('pol_num:', pol_num)

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

                #coef1 = self.integral(func_inner_prod, params) / self.integral(func_sq_norm, params)
                coef1 = self.integral(func_inner_prod, params, infty = self.is_Infty) / self.integral(func_sq_norm, params, infty = self.is_Infty)
     
                for i in range(pol_num+len_ort_polys):
                    polys_coef_matrix[pol_num+len_ort_polys, i] -= coef1 * polys_coef_matrix[it, i]

                #######################################################

                #tmp_poly = lambda x, curr_poly = tmp_poly: curr_poly(x) - (integral(func_inner_prod, _start, _stop, count) / integral(func_sq_norm, _start, _stop, count)) * ort_poly(x)
                def tmp_poly(x, cur_ort = ort_poly, curr_poly = tmp_poly, inner_prod = func_inner_prod, sq_norm = func_sq_norm, cur_coef = coef1):

                    #coef = self.integral(inner_prod, params) / self.integral(sq_norm, params)
                    return curr_poly(x) - cur_coef * cur_ort(x)


            #func_poly_norm = lambda x: tmp_poly(x) * tmp_poly(x) * measure(x)
            def func_poly_norm(x, curr_poly = tmp_poly):
                return curr_poly(x) ** 2 * measure(x)

            #tmp_poly_norm = np.sqrt(self.integral(func_poly_norm, params))
            tmp_poly_norm = np.sqrt(self.integral(func_poly_norm, params, infty = self.is_Infty))

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
        
        
        orthogonal_poly_upd = []
        
        for i in range(len(polys_to_orthogonalise) + len_ort_polys):
            
            def c_pol(x, cur_ind = i):
                res = 0
                for j in range(len(polys_to_orthogonalise) + len_ort_polys):
                    res += polys_coef_matrix[cur_ind, j] * x**j
                    
                return res
            
            orthogonal_poly_upd.append(c_pol)
        
        self.orthogonal_poly = orthogonal_poly_upd #!!!

        return orthogonal_poly_upd, polys_coef_matrix
    
    def create_orthogonal_poly_collection(self, max_poly_degrees = None):
        
        if type(max_poly_degrees) == type(None):
            max_poly_degrees = [-1] * len(self.params.keys())
        
        orthogonal_poly_collection = {}
        polys_coef_matrix_collection = {}
        
        for j in range(1, len(self.params.keys()) + 1):
            ort_pol, pol_coef_mat =  self.Gram_Smidt_process(var_index = j, max_poly_degree = max_poly_degrees[j - 1])
            
            orthogonal_poly_collection[j] = ort_pol
            polys_coef_matrix_collection[j] = pol_coef_mat
            
        self.orthogonal_poly_collection = orthogonal_poly_collection
        self.polys_coef_matrix_collection = polys_coef_matrix_collection
            
        return orthogonal_poly_collection, polys_coef_matrix_collection
    


    def create_poly_combs(self, max_pol_degree = 20):
        
        if self.orthogonal_poly_collection == None:
            self.create_orthogonal_poly_collection()
            
        orthogonal_poly_collection = self.orthogonal_poly_collection

        poly_combs = pd.DataFrame({0:[0]})

        for key in orthogonal_poly_collection.keys():
            poly_combs = poly_combs.merge(pd.DataFrame({0 : [0] * len(orthogonal_poly_collection[key]), 
                                                        key : list(range(len(orthogonal_poly_collection[key])))
                                                       }
                                                      ), 
                                          how = 'outer', 
                                          on = 0
                                         ).drop_duplicates().reset_index(drop=True)

            for j in range(len(poly_combs)):
                sum_k = sum(poly_combs.loc[j])

                if sum_k > max_pol_degree:
                    poly_combs.drop(j, axis = 0, inplace = True)
            poly_combs.reset_index(drop=True, inplace = True)

        poly_combs.drop(0, axis = 1, inplace = True)

        self.poly_combs = poly_combs
        
        return poly_combs
    
    
    def Fourier_expansion_ort(self, function):
        
        if type(self.poly_combs) == type(None):
            self.create_poly_combs()
        poly_combs = self.poly_combs
        
        if type(self.orthogonal_poly) == type(None):
            print('Orthogonal polynomials are not estimated!')
            return []
        
        ort_polys_collection = self.orthogonal_poly_collection
        polys = self.orthogonal_poly
        
        measure = self.measure

        params = self.params
        
        coef = []

        for line in tqdm(poly_combs.index):
            
            poly_set = poly_combs.loc[line]
            
            def poly_set_func(args):
                res = 1
                for key in params.keys():
                    res *= ort_polys_collection[key][poly_set[key]](args[key-1])
                return res
            
            def mes_func(args):
                res = 1
                for me in measure.keys():
                    res *= measure[me](args[me-1])
                return res
            
            def inner_product(*args):
                tmp = function(*args) * poly_set_func(args) * mes_func(args)
                return tmp
            

            coef.append(self.Fourier_integral(inner_product, params, infty = self.is_Infty))
            
        self.cur_Fourier_coefficients = coef

        return coef


    def Fourier_expansion(self, function, max_poly_degree = -1):
        
        if max_poly_degree == -1:
            max_poly_degree = self.max_poly_degree
            
        print('Max Polynomial degree:', max_poly_degree)
        
        measure = self.measure
        params = self.params

        orthogonal_poly, polys_coef_matrix = self.create_orthogonal_poly_collection(max_poly_degrees = [max_poly_degree] * len(params.keys()))

        coef = self.Fourier_expansion_ort(function)
        
        self.cur_Fourier_coefficients = coef

        num_var = len(params.keys())
        
        vars_ = [0]
        
        for j in range(1, num_var + 1):
            vars_.append(symbols('x'+str(j)))
            
        self._vars = vars_
            
        res = 0
        
        for line in tqdm(self.poly_combs.index):
            tmp = coef[line]
            poly_set = self.poly_combs.loc[line]
            for key in params.keys():
                tmp *= orthogonal_poly[key][poly_set[key]](vars_[key])

            res += tmp
            
        res = res.as_poly()
            
        self.cur_mono_coefficients = res

        return res
    
    def error_estimator(self, function):
        
        params = self.params

        
        def numerator(*args):
            y = self.cur_mono_coefficients
            m = 1
            for i in range(len(args)):
                y = y.subs({self._vars[i + 1] : args[i]})
                m *= self.measure[i + 1](args[i])
            
            tmp_res = (function(*args) - y)**2 * m
            
            return tmp_res #np.where(np.isnan(tmp_res), 0, tmp_res)+0
        
        
        res = self.Fourier_integral(numerator, params, infty = self.is_Infty)
        
        
        return np.sqrt(res)
