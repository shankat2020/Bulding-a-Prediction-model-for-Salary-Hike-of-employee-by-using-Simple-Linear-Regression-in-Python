##############################################################################
###################### Simple Linear Regression ##############################
##############################################################################






#4) Salary_hike -> Build a prediction model for Salary_hike

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
#EDA(explotary data analysis)
Salary_Data=pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Simple_linear_Regression\\Salary_Data.csv")
Salary_Data.columns




######### EDA(explotary data analysis) #################

######### 1st moment business decision ###########
Salary_Data.mean()#YearsExperience        5.313333
                   # Salary             76003.000000
Salary_Data.median()#YearsExperience        4.7
                     #Salary             65237.0
Salary_Data.mode()



##########2nd moment busines decision ##############

Salary_Data.var()  #YearsExperience    8.053609e+00
                    #Salary             7.515510e+08                   
Salary_Data.std() #YearsExperience        2.837888
                  #Salary             27414.429785                  

max(Salary_Data['Salary'])#122391
max(Salary_Data['YearsExperience'])#10.5
Range=max(Salary_Data['Salary'])-min(Salary_Data['Salary'])


########### 3rd and 4th moment business decision #########

Salary_Data.skew()#YearsExperience    0.37956
                  #Salary             0.35412# both the data are positevely skewed

Salary_Data.kurt()#YearsExperience   -1.012212
                  #Salary            -1.295421# since the kurtosis value is negative
                                              #implies both the distribution have wider peaks

#### Graphical representation   #########
                  
plt.hist(Salary_Data.YearsExperience)
plt.boxplot(Salary_Data.YearsExperience,0,"rs",0)


plt.hist(Salary_Data.Salary)
plt.boxplot(Salary_Data.Salary)

plt.plot(Salary_Data.YearsExperience,Salary_Data.Salary,"bo");plt.xlabel("Years_Of_Exprience");plt.ylabel("Salary")


Salary_Data.Salary.corr(Salary_Data.YearsExperience) # 0.9782416184887598 # correlation value between X and Y

### or ### table format
Salary_Data.corr()           #                YearsExperience    Salary
#                          YearsExperience         1.000000     0.978242
 #                          Salary                  0.978242     1.000000
                  

#or using numpy
np.corrcoef(Salary_Data.YearsExperience,Salary_Data.Salary)

import seaborn as sns
sns.pairplot(Salary_Data)




############## Model Preparing/ injecting the model #################



# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=Salary_Data).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()#Adj. R-squared:                  0.955

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(Salary_Data.iloc[:,0]) # Predicted values of Salary using the model
 



# Visualization of regresion line over the scatter plot of YearsExperience and Salary
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=Salary_Data['YearsExperience'],y=Salary_Data['Salary'],color='red');plt.plot(Salary_Data['YearsExperience'],pred,color='black');plt.xlabel('YrsExp');plt.ylabel('SALARY')

pred.corr(Salary_Data.Salary) # 0.9782416184887599
# Predicted vs actual values
plt.scatter(x=pred,y=Salary_Data.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")




# Transforming variables for accuracy
model2 = smf.ols('Salary~np.log(YearsExperience)',data=Salary_Data).fit()
model2.params#Intercept                  14927.97177
             #np.log(YearsExperience)    40581.98796
model2.summary()#Adj. R-squared:                  0.849

print(model2.conf_int(0.01)) # 99% confidence level

pred2 = model2.predict(pd.DataFrame(Salary_Data['YearsExperience']))

pred2.corr(Salary_Data.Salary)#0.9240610817882641
pred21 = model2.predict(Salary_Data.iloc[:,0])
pred21
plt.scatter(x=Salary_Data['YearsExperience'],y=Salary_Data['Salary'],color='green');plt.plot(Salary_Data['YearsExperience'],pred21,color='blue');plt.xlabel('YearsExp');plt.ylabel('Salary')




# Exponential transformation
model3 = smf.ols('np.log(Salary)~YearsExperience',data=Salary_Data).fit()
model3.params
model3.summary()# 0.930
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(Salary_Data['YearsExperience']))
pred_log
pred3=np.exp(pred_log)  
pred3
pred3.corr(Salary_Data.Salary)#0.9660469705377088
plt.scatter(x=Salary_Data['YearsExperience'],y=Salary_Data['Salary'],color='green');plt.plot(Salary_Data.YearsExperience,np.exp(pred_log),color='blue');plt.xlabel('YearsExp');plt.ylabel('SALARY')
resid_3 = pred3-Salary_Data.Salary#error




# so we will consider the model having highest R-Squared value which is the 1st  model
# getting residuals of the entire data set
Salary_Data_resid = model.resid_pearson #error
Salary_Data_resid 
plt.plot(model.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred,y=Salary_Data.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")


#we can also check the other transformation 
# Quadratic model
#Salary_Data["YearsExperience_Sq"] = Salary_Data.YearsExperience*Salary_Data.YearsExperience
#Salary_Data#1 extra column will be formed
model_quad = smf.ols("Salary~YearsExperience+YearsExperience*YearsExperience",data=Salary_Data).fit()
model_quad.params#Intercept          25792.200199
                 #YearsExperience     9449.962321
model_quad.summary()#Adj. R-squared:                  0.954
pred_quad = model_quad.predict(Salary_Data.YearsExperience)

model_quad.conf_int(0.05) # 
plt.scatter(Salary_Data.YearsExperience,Salary_Data.Salary,c="b");plt.plot(Salary_Data.YearsExperience,pred_quad,"r")

plt.scatter(np.arange(30),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 


## so we can c without any transformation, the 1st model is giving accuracy having the highest R-squared value












