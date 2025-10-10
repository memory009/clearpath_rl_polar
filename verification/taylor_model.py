#!/usr/bin/env python3
"""
Taylor模型和POLAR可达性验证模块
v8
"""

import numpy as np
import sympy as sym
from functools import reduce
import operator as op
import math
import sys
import os

# 添加当前目录到路径，解决导入问题
sys.path.append(os.path.dirname(__file__))


def ncr(n, r):
    """组合数 C(n,r)"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


class TaylorModel:
    """Taylor模型：多项式 + 误差区间"""
    def __init__(self, poly, remainder):
        self.poly = poly
        self.remainder = remainder


class TaylorArithmetic:
    """Taylor算术运算"""
    
    def weighted_sumforall(self, taylor_models, weights, bias):
        """计算加权和：Σ(w_i * TM_i) + bias"""
        temp_poly = 0
        for i, tm in enumerate(taylor_models):
            temp_poly += weights[i] * tm.poly
        temp_poly += bias
        
        temp_remainder = 0
        for i, tm in enumerate(taylor_models):
            temp_remainder += abs(weights[i]) * tm.remainder[1]
        
        remainder = [-temp_remainder, temp_remainder]
        
        # 修复：确保 temp_poly 是 Poly 对象
        if not isinstance(temp_poly, sym.Poly):
            # 如果是常数，需要创建带生成器的 Poly
            if hasattr(taylor_models[0].poly, 'gens') and taylor_models[0].poly.gens:
                temp_poly = sym.Poly(temp_poly, *taylor_models[0].poly.gens)
            else:
                temp_poly = sym.Poly(temp_poly)
        
        return TaylorModel(temp_poly, remainder)
    
    def constant_product(self, taylor_model, constant):
        """常数乘法：c * TM"""
        new_poly = constant * taylor_model.poly
        new_remainder = [constant * taylor_model.remainder[0],
                        constant * taylor_model.remainder[1]]
        
        # 修复：确保结果是 Poly 对象
        if not isinstance(new_poly, sym.Poly):
            if hasattr(taylor_model.poly, 'gens') and taylor_model.poly.gens:
                new_poly = sym.Poly(new_poly, *taylor_model.poly.gens)
            else:
                new_poly = sym.Poly(new_poly)
        
        return TaylorModel(new_poly, new_remainder)


class BernsteinPolynomial:
    """Bernstein多项式逼近"""
    
    def __init__(self, error_steps=4000):
        self.error_steps = error_steps
        self.bern_poly = None
    
    def approximate(self, a, b, order, activation_name):
        """在区间[a, b]上用Bernstein多项式逼近激活函数"""
        # 修复导入问题
        try:
            from activation_functions import Activation_functions
        except ImportError:
            from verification.activation_functions import Activation_functions
        
        d_max = 8
        d_p = np.floor(d_max / math.log10(1 / (b - a)))
        d_p = np.abs(d_p)
        if d_p < 2:
            d_p = 2
        d = min(order, d_p)
        
        coe1_1 = -a / (b - a)
        coe1_2 = 1 / (b - a)
        coe2_1 = b / (b - a)
        coe2_2 = -1 / (b - a)
        
        x = sym.Symbol('x')
        bern_poly = 0 * x
        
        for v in range(d + 1):
            c = ncr(d, v)
            point = a + (b - a) / d * v
            
            if activation_name == 'relu':
                f_value = Activation_functions.relu(point)
            elif activation_name == 'tanh':
                f_value = Activation_functions.tanh(point)
            else:
                raise ValueError(f"不支持的激活函数: {activation_name}")
            
            basis = ((coe1_2 * x + coe1_1) ** v * 
                    (coe2_1 + coe2_2 * x) ** (d - v))
            
            bern_poly += c * f_value * basis
        
        if bern_poly == 0:
            bern_poly = 1e-16 * x
        
        self.bern_poly = bern_poly
        return sym.Poly(bern_poly)
    
    def compute_error(self, a, b, activation_name):
        """计算Bernstein逼近的误差上界"""
        # 修复导入问题
        try:
            from activation_functions import Activation_functions
        except ImportError:
            from verification.activation_functions import Activation_functions
        
        epsilon = 0
        m = self.error_steps
        
        for v in range(m + 1):
            point = a + (b - a) / m * (v + 0.5)
            
            if activation_name == 'relu':
                f_value = Activation_functions.relu(point)
            elif activation_name == 'tanh':
                f_value = Activation_functions.tanh(point)
            
            b_value = sym.Poly(self.bern_poly).eval(point)
            temp_diff = abs(f_value - b_value)
            epsilon = max(epsilon, temp_diff)
        
        return epsilon + (b - a) / m


def compute_tm_bounds(tm):
    """计算Taylor模型的上下界"""
    poly = tm.poly
    
    temp_upper = 0
    temp_lower = 0
    
    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        
        if sum(monom) == 0:
            temp_upper += coeff
            temp_lower += coeff
        else:
            temp_upper += abs(coeff)
            temp_lower += -abs(coeff)
    
    a = temp_lower + tm.remainder[0]
    b = temp_upper + tm.remainder[1]
    
    # 转换为Python原生float类型
    return float(a), float(b)


def compute_poly_bounds(poly):
    """计算多项式的上下界"""
    temp_upper = 0
    temp_lower = 0
    
    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        
        if sum(monom) == 0:
            temp_upper += coeff
            temp_lower += coeff
        else:
            temp_upper += abs(coeff)
            temp_lower += -abs(coeff)
    
    # 转换为Python原生float类型
    return float(temp_lower), float(temp_upper)


def apply_activation(tm, bern_poly, bern_error, max_order):
    """通过Bernstein多项式传播Taylor模型过激活函数"""
    # 1. 合成多项式
    composed = sym.polys.polytools.compose(bern_poly, tm.poly)
    
    # 2. 截断到指定阶数
    poly_truncated = 0
    for i in range(len(composed.monoms())):
        monom = composed.monoms()[i]
        if sum(monom) <= max_order:
            temp = 1
            for j in range(len(monom)):
                temp *= composed.gens[j] ** monom[j]
            poly_truncated += composed.coeffs()[i] * temp
    
    poly_truncated = sym.Poly(poly_truncated)
    
    # 3. 计算截断误差
    poly_remainder = composed - poly_truncated
    _, truncation_error = compute_poly_bounds(poly_remainder)
    
    # 4. 计算总误差
    total_remainder = 0
    
    for i in range(len(bern_poly.monoms())):
        monom = bern_poly.monoms()[i]
        degree = sum(monom)
        
        if degree < 1:
            continue
        elif degree == 1:
            total_remainder += abs(bern_poly.coeffs()[i] * tm.remainder[1])
        else:
            total_remainder += abs(bern_poly.coeffs()[i] * (tm.remainder[1] ** degree))
    
    remainder = [
        -total_remainder - truncation_error - bern_error,
        total_remainder + truncation_error + bern_error
    ]
    
    return TaylorModel(poly_truncated, remainder)