from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import style
style.use('fivethirtyeight')

#In this function random data is created.You can also use CSV file. In this function hm stands how many variables you want in your dataset , variance is used to decide
#the range of variables, step is used to increment and corr stands for correlation.Correlation is used to create positive or negative y axis.

def create_dataset(hm,variance, step=2, corr=False):
    val=1
    ys=[]
    for i in range(hm):
        y= val + random.randrange(-variance, variance)  # Initially y is equal to val that is 1.
        ys.append(y)
        if corr and corr=='pos':
            val+=step
        elif corr and corr=='neg':
            val-= step
    xs=[i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):      # this function returns the values of m(slope) and intercept(b).
    m= (  ((mean(xs)*mean(ys)) - mean(xs*ys)) /
        ((mean(xs)**2) - mean(xs**2))  )
    b= mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orig, ys_line):   #This function is used to determine the squared error(sum of square of residuals(ys-y^)**2 and total sum of squares(ys-mean(ys)**2)
    return sum((ys_line-ys_orig)**2)

def coff_of_determination(ys_orig,ys_line):     # This function is used to determine the r_squared value.
    y_mean_line=[mean(ys_orig)]*len(ys_orig)                         
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1- (squared_error_regr/squared_error_y_mean)


xs,ys=create_dataset(40,10,2,corr='pos')   # Smaller the value of variance, better be the r_squared value which tells us about how good is your predicted line as
m,b =best_fit_slope_and_intercept(xs,ys)   # compared  to the average line. In other words it determines goodness of the fit of the model.

regression_line=[(m*x)+b for x in xs]     #Here we are creating our regreesion line(y^)

predict_x= 33
predict_y= (m*predict_x)+b

r_squared= coff_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys,color='red')               #In this section we are visualising our results.
plt.plot(xs,regression_line,color='blue')
plt.scatter(predict_x,predict_y,s=100,color='green')
plt.show()
