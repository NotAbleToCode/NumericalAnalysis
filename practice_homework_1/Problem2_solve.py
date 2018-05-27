import numpy as np
import matplotlib.pyplot as plt
import math
def my_fun(x):
    return 1/(1 + x**2)

#梯形公式，number等分
def trapezoid_formula(my_fun, bottom, top, number):
    total_sum = 0
    step = (top - bottom)/number
    total_sum += my_fun(bottom)
    if number >= 2:
        for i in range(number - 1):
            total_sum += (2 * my_fun(bottom + (i+1) * step))
    total_sum += my_fun(top)
    return step*total_sum/2

#辛普森公式，number等分t
def Simpson_formula(my_fun, bottom, top, number):
    total_sum1 = 0
    total_sum2 = 0
    step = (top - bottom)/number
    if number >= 2:
        for i in range(number - 1):
            total_sum1 += (2 * my_fun(bottom + (i+1) * step))
    if number >= 1:
        for i in range(number):
            total_sum2 += (4 * my_fun(bottom + i * step + step/2))
    return step*(my_fun(bottom) + total_sum1 + total_sum2 + my_fun(top))/6

#对零点和系数，可以查表得到
#两点求积：高斯点-0.5773503和0.5773503，系数为1,1
#三点求积：高斯点-0.7745967，0，0.7745967，系数为0.5555556，0.8888889，0.5555556
#五点求积：高斯点-0.9061798，-0.5384693，0，0.5384693，0.9061798
#系数为0.2369269，0.4786287，0.5688889，0.4786287，0.2369269

#复合两点高斯求积，number等分
def Gauss_formula_2(my_fun, bottom, top, number):
    step = (top - bottom)/number
    value = 0
    transform = lambda x,sub_bott,sub_top: x*(sub_top-sub_bott)/2+(sub_bott+sub_top)/2 
    for i in range(number):
        sub_bott = bottom + i*step 
        sub_top = sub_bott + step
        value += (my_fun(transform(-0.5773503,sub_bott,sub_top)) + my_fun(transform(0.5773503,sub_bott,sub_top)))
    return value*step/2


#复合三点高斯求积，number等分
def Gauss_formula_3(my_fun, bottom, top, number):
    step = (top - bottom)/number
    value = 0
    transform = lambda x,sub_bott,sub_top: x*(sub_top-sub_bott)/2+(sub_bott+sub_top)/2 
    for i in range(number):
        sub_bott = bottom + i*step 
        sub_top = sub_bott + step
        value += (0.5555556*my_fun(transform(-0.7745967,sub_bott,sub_top)) 
        + 0.8888889*my_fun(transform(0,sub_bott,sub_top)) 
        + 0.5555556*my_fun(transform(0.7745967,sub_bott,sub_top)))
    return value*step/2

#复合五点高斯求积，number等分
#五点求积：高斯点-0.9061798，-0.5384693，0，0.5384693，0.9061798
#系数为0.2369269，0.4786287，0.5688889，0.4786287，0.2369269
def Gauss_formula_5(my_fun, bottom, top, number):
    step = (top - bottom)/number
    value = 0
    transform = lambda x,sub_bott,sub_top: x*(sub_top-sub_bott)/2+(sub_bott+sub_top)/2
    for i in range(number):
        sub_bott = bottom + i*step 
        sub_top = sub_bott + step
        value += (0.2369269*my_fun(transform(-0.9061798,sub_bott,sub_top)) 
        + 0.4786287*my_fun(transform(-0.5384693,sub_bott,sub_top)) 
        + 0.5688889*my_fun(transform(0,sub_bott,sub_top))
        + 0.4786287*my_fun(transform(0.5384693,sub_bott,sub_top))
        + 0.2369269*my_fun(transform(0.9061798,sub_bott,sub_top)))
    return value*step/2

def error_estimate_trapezoid_formula(fun, n):
    #等分的区间数、
    number_list = [i for i in range(70,80)]
    #对应的误差
    error_list = [abs(fun(my_fun, -1, 1, n) - math.pi/2) for n in number_list] 
    #对应的区间长度
    step_list = [(2/i)**n for i in number_list]
    plt.plot(step_list, error_list)
    plt.show()

if __name__ == '__main__':
    print('精确积分：', math.pi/2)
    print('梯形公式：', trapezoid_formula(my_fun, -1, 1, 100))
    print('辛普森公式：', Simpson_formula(my_fun, -1, 1, 10))
    print('两点高斯公式：', Gauss_formula_2(my_fun, -1, 1, 200))
    print('三点高斯公式：', Gauss_formula_3(my_fun, -1, 1, 200))
    print('五点高斯公式：', Gauss_formula_5(my_fun, -1, 1, 4))
    error_estimate_trapezoid_formula(trapezoid_formula, 2)
    error_estimate_trapezoid_formula(Simpson_formula, 4)
    error_estimate_trapezoid_formula(Gauss_formula_2, 4)
    error_estimate_trapezoid_formula(Gauss_formula_3, 6)
    error_estimate_trapezoid_formula(Gauss_formula_5, 10)
    


