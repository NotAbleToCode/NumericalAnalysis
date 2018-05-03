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
def hermite_interpolation():
    pass

#分段线性插值
def piecewise_liner_interpolation(my_fun, point_x):
    def piecewise_liner_fun(x):
        bottom = math.floor(x)
        top = bottom + 1
        return (my_fun(top) - my_fun(bottom))*(x - bottom)/(top - bottom) + my_fun(bottom)
    return piecewise_liner_fun

#分段埃尔米特插值
def piecewise_hermite_interpolation(my_fun, my_fun_der, point_x):
    def piecewise_hermite_fun(x):
        bottom = math.floor(x)
        top = bottom + 1
        return (((x - top)/(bottom - top))**2 * (1 + 2 * (x - bottom)/(top - bottom)) * my_fun(bottom)
        + ((x - bottom)/(top - bottom))**2 * (1 + 2 * (x - top)/(bottom - top)) * my_fun(top)
        + ((x - top)/(bottom - top))**2 * (x - bottom) * my_fun_der(bottom)
        + ((x - bottom)/(top - bottom))**2 * (x - top) * my_fun_der(top))
    return piecewise_hermite_fun

#三次样条插值
def three_spline_interpolation(my_fun, my_fun_der, point_x):
    pass

#展示原函数和插值函数
def display_fun(original_fun, inter_fun, bottom, top):
    x = np.linspace(bottom, top, 1000)  #这个表示在bottom到top之间生成1000个x值
    y1 = [ original_fun(i) for i in x] 
    y2 = [ inter_fun(i) for i in x]  
    plt.figure('Display Of Function Image') #创建一个窗口
    plt.subplot(1, 3, 1) #将该窗口分为一行三列，并定位到第一列
    plt.title('Original Function')
    plt.plot(x, y1) #当前定位绘制图像
    plt.subplot(1, 3, 2) #定位到第二列
    plt.title(inter_fun.__name__)
    plt.plot(x, y2) #绘制图像
    plt.subplot(1, 3, 3)
    plt.title('Contrast Image')
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show() #展示所有窗口

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
    number = 100
    point_x = [i for i in range(bottom, top + 1)]
    lagrange_fun = lagrange_interpolation(my_fun, point_x)
    piecewise_liner_fun = piecewise_liner_interpolation(my_fun, point_x)
    piecewise_hermite_fun = piecewise_hermite_interpolation(my_fun, my_fun_der, point_x)
    #display_fun(my_fun, lagrange_fun, bottom, top)
    #display_fun(my_fun, piecewise_liner_fun, bottom, top)
    #display_fun(my_fun, piecewise_hermite_fun, bottom, top)
    print('The mean error is as follows:')
    print('Lagrange_fun:', average_error(my_fun, lagrange_fun, bottom, top, number))
    print('piecewise_liner_fun:', average_error(my_fun, piecewise_liner_fun, bottom, top, number))
    print('piecewise_hermite_fun:', average_error(my_fun, piecewise_hermite_fun, bottom, top, number))
    

