import numpy as np
import matplotlib.pyplot as plt
import math

#要插值的函数
def my_fun(x):
    return 1/(1 + x**2)

#要插值的函数的导数
def my_fun_der(x):
    return -2 * x /(1 + x**2)**2

#拉格朗日插值
def lagrange_interpolation(my_fun, point_x):
    point_y = [my_fun(i) for i in point_x]
    point_x_y = list(zip(point_x ,point_y))
    def lagrange_fun(x):#定义拉格朗日插值函数
        sum = 0
        for point_k in point_x_y:
            top = 1#每个插值基函数的分子的值
            bottom = 1#每个插值基函数的分母的值
            for point_i in point_x_y:
                if point_i[0] != point_k[0]:
                    top *= (x - point_i[0])
                    bottom *= (point_k[0] - point_i[0])
            sum += top * point_k[1] / bottom
        return sum
    return lagrange_fun#返回插值得到的函数

    
#埃尔米特插值
def hermite_interpolation(my_fun, my_fun_der, point_x):
    def hermite_fun(x):
        function_value = 0
        for base_x in point_x:
            multi1 = 1
            multi2 = 1
            l_der = 0
            for factor in point_x:
                if factor != base_x:
                    multi1 *= (x - factor)
                    multi2 *= (base_x - factor)
                    l_der += 1/(base_x - factor)
            l_square = (multi1/multi2)**2
            function_value += (my_fun(base_x)*(1 - 2*(x - base_x)*l_der)*l_square + my_fun_der(base_x)*(x - base_x)*l_square)
        return function_value
    return hermite_fun


#分段线性插值，要求插值点间隔为1等距分布
def piecewise_liner_interpolation(my_fun, point_x):
    def piecewise_liner_fun(x):
        bottom = math.floor(x)
        top = bottom + 1
        return (my_fun(top) - my_fun(bottom))*(x - bottom)/(top - bottom) + my_fun(bottom)
    return piecewise_liner_fun

#分段埃尔米特插值，要求插值点间隔为1等距分布
def piecewise_hermite_interpolation(my_fun, my_fun_der, point_x):
    def piecewise_hermite_fun(x):
        bottom = math.floor(x)
        top = bottom + 1
        return (((x - top)/(bottom - top))**2 * (1 + 2 * (x - bottom)/(top - bottom)) * my_fun(bottom)
        + ((x - bottom)/(top - bottom))**2 * (1 + 2 * (x - top)/(bottom - top)) * my_fun(top)
        + ((x - top)/(bottom - top))**2 * (x - bottom) * my_fun_der(bottom)
        + ((x - bottom)/(top - bottom))**2 * (x - top) * my_fun_der(top))
    return piecewise_hermite_fun

#三次样条插值，取边界条件为边界点一阶导数为1,要求插值点间隔为1等距分布
def three_spline_interpolation(my_fun, point_x, der_value_left, der_value_right):
    list_a = [0]
    list_b = [1]
    value_list = []
    for index in range(point_x.__len__() - 2):
        list_a.append((point_x[index + 1] - point_x[index])/(point_x[index + 2] - point_x[index]))
        list_b.append((point_x[index + 2] - point_x[index + 1])/(point_x[index + 2] - point_x[index]))
    list_a.append(1)
    list_b.append(0)
    list_2 = [2]*point_x.__len__()
    coe_mat = list(zip(list_a, list_2, list_b))#系数矩阵
    coe_mat = [list(i) for i in coe_mat]
    d = 6*(my_fun(point_x[0]) - my_fun(point_x[1]) - der_value_left)
    value_list.append(d)
    for index in range(point_x.__len__() - 2):
        d = 3 * (my_fun(point_x[index + 2]) - my_fun(point_x[index + 1]) - my_fun(point_x[index + 1]) + my_fun(point_x[index ]))
        value_list.append(d)
    d = 6 * (der_value_right - my_fun(point_x[-1]) + my_fun(point_x[-2]))
    value_list.append(d)
    solve_list = three_diagonal_matrices_Gauss_solve(coe_mat, value_list)
    def three_spline_fun(x):
        bottom = math.floor(x)
        top = bottom + 1
        if bottom >= point_x[-1]:#x值为右边界的情况
            return my_fun(x)
        M0 = solve_list[point_x.index(bottom)]
        M1 = solve_list[point_x.index(top)]
        value = M0 * (top - x)**3 / 6 + M1 * (x - bottom)**3 / 6 + (my_fun(bottom) - M0/6)*(top - x) + (my_fun(top) - M1/6)*(x - bottom)
        return value
    return three_spline_fun



#三对角矩阵的高斯消元法求解
#coe_mat为一个n×3的矩阵，存储三对角矩阵的元素值，其中默认coe_mat[0][0] = coe_mat[n-1][2] =0
def three_diagonal_matrices_Gauss_solve(coe_mat, value_list):
    solve_list = []
    for index_row in range(coe_mat.__len__() - 1):
        factor = - coe_mat[index_row + 1][0] / coe_mat[index_row][1]
        coe_mat[index_row + 1][0] += factor * coe_mat[index_row][1]
        coe_mat[index_row + 1][1] += factor * coe_mat[index_row][2]
        value_list[index_row + 1] += factor * value_list[index_row]
    #print(coe_mat)
    solve_list.append(value_list[-1]/coe_mat[-1][1])#求出xn
    #向前迭代求xn-1,xn-2...
    for index_row in range(coe_mat.__len__() - 1):
        index = -index_row -2 #从后向前查找
        solve_list.append((value_list[index] - coe_mat[index][2] * solve_list[-1])/ coe_mat[index][1])
    solve_list.reverse()
    return solve_list

#生成区间[bottom, top]上n个切比雪夫多项式零点
def Chebyshev_polynomials_zero_point(bottom, top, n):
    zero_list = []
    for i in range(n):
        temp = math.cos((2*i+1)/(2*n)*math.pi)
        zero_list.append((top - bottom) * (temp + 1) / 2 + bottom)
    return zero_list

#展示原函数和插值函数
def display_fun(original_fun, inter_fun, bottom, top, name):
    x = np.linspace(bottom, top, 1000)  #这个表示在bottom到top之间生成1000个x值
    y1 = [ original_fun(i) for i in x] 
    y2 = [ inter_fun(i) for i in x]  
    plt.figure('Display Of Function Image: ' + name) #创建一个窗口
    plt.subplot(1, 3, 1) #将该窗口分为一行三列，并定位到第一列
    plt.title('Original Function')
    plt.plot(x, y1) #当前定位绘制图像
    plt.subplot(1, 3, 2) #定位到第二列
    plt.title(name)
    plt.plot(x, y2) #绘制图像
    plt.subplot(1, 3, 3)
    plt.title('Contrast Image')
    plt.plot(x, y1)
    plt.plot(x, y2)

#计算原函数和插值函数的误差
#number是等距分成的小区间数
def average_error(original_fun, inter_fun, bottom, top, number):
    step = (top - bottom) / 100
    current = bottom
    total_error = 0
    while current <= top:
        total_error += abs(original_fun(current) - inter_fun(current))
        current += step
    #top点处的误差，虽然肯定为零，形式上还是写出来吧，(o.o)
    total_error += abs(original_fun(current) - inter_fun(current))
    return total_error/(number + 1)

if __name__ == '__main__':
    bottom = -5
    top = 5
    number = 100 #取number个等距点进行误差的均值计算
    point_x = [i for i in range(bottom, top + 1)]
    lagrange_fun = lagrange_interpolation(my_fun, point_x)
    hermite_fun = hermite_interpolation(my_fun, my_fun_der, point_x)
    #lagrange_fun_Chebyshev = lagrange_interpolation(my_fun, Chebyshev_polynomials_zero_point(bottom, top, top - bottom + 1))
    #piecewise_liner_fun = piecewise_liner_interpolation(my_fun, point_x)
    #piecewise_hermite_fun = piecewise_hermite_interpolation(my_fun, my_fun_der, point_x)
    #three_spline_fun = three_spline_interpolation(my_fun, point_x, 1, 1)
    display_fun(my_fun, lagrange_fun, bottom, top, 'lagrange_fun')
    display_fun(my_fun, hermite_fun, bottom, top, 'hermite_fun')
    #display_fun(my_fun, lagrange_fun_Chebyshev, bottom, top, 'lagrange_fun_Chebyshev')
    #display_fun(my_fun, piecewise_liner_fun, bottom, top, 'piecewise_liner_fun')
    #display_fun(my_fun, piecewise_hermite_fun, bottom, top, 'piecewise_hermite_fun')
    #display_fun(my_fun, three_spline_fun, bottom, top, 'three_spline_fun')
    #print('The mean error is as follows:')
    print('lagrange_fun:', average_error(my_fun, lagrange_fun, bottom, top, number))
    print('hermite_fun:', average_error(my_fun, hermite_fun, bottom, top, number))
    #print('lagrange_fun_Chebyshev:', average_error(my_fun, lagrange_fun_Chebyshev, bottom, top, number))
    #print('piecewise_liner_fun:', average_error(my_fun, piecewise_liner_fun, bottom, top, number))
    #print('piecewise_hermite_fun:', average_error(my_fun, piecewise_hermite_fun, bottom, top, number))
    #print('three_spline_fun:', average_error(my_fun, three_spline_fun, bottom, top, number))
    plt.show() #展示所有窗口
    

    