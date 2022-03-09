
import scipy.io
import numpy as np
train_data = scipy.io.loadmat("train_data.mat")
test_data = scipy.io.loadmat("test_data.mat")

#*********************************************************************************************************************
'''
Task-1 : Feature extraction and Normalisation
'''
#*********************************************************************************************************************



#From the data dictionary return by the matlab file we are acccesing the data with key 'data'
train_d = train_data['data']

# creating list elements for calculating and storing means and standard deviation of each image
train_mean_img=[]
train_sd_img=[]

for i in range(len(train_d)):
    train_mean_img.append(train_d[i].mean())
    train_sd_img.append(train_d[i].std())


# converting list to numpy arrays to apply mean and std functions
train_mean_img=np.array(train_mean_img)
train_sd_img=np.array(train_sd_img)

# printing the means and std of each image
print(f"The feature representations for each digit mean: {train_mean_img} , standard deviation: {train_sd_img}")
'''
    1. calculating the means and standard deviations specific to the training data
    2. we use this data to normalise the feature vectors of traing and testing data
'''
# calculating Means and Std specific to all indiavidual means and individual standard deviation
M_mean = train_mean_img.mean()
S_mean = train_mean_img.std()
M_sd = train_sd_img.mean()
S_sd = train_sd_img.std()

print(f"M1: {M_mean}, S1: {S_mean}")
print(f"M2: {M_sd}, S2: {S_sd}")


#####################'''feature vector calculation for traing data'''##############################
train_Y_mean=[]
train_Y_sd=[]
for i in range(len(train_d)):
    train_Y_mean.append((train_mean_img[i]-M_mean)/S_mean)
    train_Y_sd.append((train_sd_img[i]-M_sd)/S_sd)

# Calculationg the normalised means and standard deviation of all the individual images using list comprehension
train_Y_img=np.transpose([[train_Y_mean[i],train_Y_sd[i]] for i in range(len(train_d))])
print(train_Y_img)

print(f"Feature Vector of {len(train_Y_img[0])} training samples is : {train_Y_img}")



#####################'''feature vector calculation for testing data'''##############################

#From the data dictionary return by the matlab file we are acccesing the data with key 'data'
test_d = test_data['data']

# creating list elements for calculating and storing means and standard deviation of each image
test_mean_img=[]
test_sd_img=[]

for i in range(len(test_d)):
    test_mean_img.append(test_d[i].mean())
    test_sd_img.append(test_d[i].std())

# converting list to numpy arrays to apply mean and std functions
test_mean_img=np.array(test_mean_img)
test_sd_img=np.array(test_sd_img)

print(f"The feature representations for each digit mean: {test_mean_img} , standard deviation: {test_sd_img}")
'''
   we use the same M1, M2, S1, and S2 values generated for the training samples on testing data
   to normalise the feature vector of testing.
'''

test_Y_mean=[]
test_Y_sd=[]
for i in range(len(test_d)):
    test_Y_mean.append((test_mean_img[i]-M_mean)/S_mean)
    test_Y_sd.append((test_sd_img[i]-M_sd)/S_sd)

# Calculationg the normalised means and standard deviation of all the individual images
test_Y_img=np.transpose([[test_Y_mean[i],test_Y_sd[i]] for i in range(len(test_d))])

print(f"feature vector of {len(test_Y_img[0])} testing samples is : {test_Y_img} ")
#*********************************************************************************************************************
'''
Task-2 : Density Estimation
'''
#*********************************************************************************************************************

# Calculating the individual means for class/digit 3 and 7
# creating two new variables to calculate means of 3 and 7 seperatly
# we know there are 5713 '3's and 5835 '7's

mean_3 = train_Y_img[:, 0:5712].mean(axis=1)
mean_7 = train_Y_img[:, 5713:].mean(axis=1)

print(f"mean vector of class 3 is {mean_3}")
print(f"mean vector of class 7 is {mean_7}")

# covarience matrix for the class/digit 3 and 7 respectivly
cov_3 = np.cov(train_Y_img[:, 0:5712])
cov_7 = np.cov(train_Y_img[:, 5713:])

print(f"covarience matrix of class 3 is {cov_3}")
print(f"covarience matrix of class 7 is {cov_7}")




#*********************************************************************************************************************
'''
Task-3 : Bayesian Decision Theory for optimal classification

Calculate probability of error for the following cases for both training sets and testing sets

case1. prior probabilities are P(3) = p(7) = 0.5 
case2. prior probabilities are p(3) = 0.3 and p(7) = 0.7
'''
#*********************************************************************************************************************
# calculating errors for each image
training_data_error=[]
testing_data_error=[]

def Calc_Likelihood(x_vec, mu, cov, d ):
    A = 1 / (pow((2 * np.pi),d/2) * (pow(np.linalg.det(cov), 1/2)))
    exponent = (-(1/2) * np.matmul(np.transpose(x_vec - mu), np.matmul(np.linalg.inv(cov), x_vec - mu)))
    return A * np.exp(exponent)

############################# Probability of error for training data ################################################

def probability_of_error_traing_data(prior_3, prior_7):
    for i in range(len(train_Y_img[0])):
        likelihood_3 = Calc_Likelihood(train_Y_img[:, i], mean_3, cov_3, len(train_Y_img[:, i]))
        likelihood_7 = Calc_Likelihood(train_Y_img[:, i], mean_7, cov_7, len(train_Y_img[:, i]))

        # evidence = likelihood_3 * prior_3 + likelihood_7 * prior_7
        evidence = likelihood_3 * prior_3 + likelihood_7 * prior_7

        probability_3 = (likelihood_3 * prior_3 ) / evidence
        probability_7 = (likelihood_7 * prior_7) / evidence

        if probability_3>probability_7:
            training_data_error.append(probability_7)
        else:
            training_data_error.append(probability_3)
    return training_data_error

############################# Probability of error for testing data ################################################

def probability_of_error_testing_data(prior_3, prior_7):
    for i in range(len(test_Y_img[0])):
        likelihood_3 = Calc_Likelihood(test_Y_img[:, i], mean_3, cov_3, len(test_Y_img[:, i]))
        likelihood_7 = Calc_Likelihood(test_Y_img[:, i], mean_7, cov_7, len(test_Y_img[:, i]))

        # evidence = likelihood_3 * prior_3 + likelihood_7 * prior_7
        evidence = likelihood_3 * prior_3 + likelihood_7 * prior_7

        probability_3 = (likelihood_3 * prior_3) / evidence
        probability_7 = (likelihood_7 * prior_7) / evidence

        if probability_3 > probability_7:
            testing_data_error.append(probability_7)
        else:
            testing_data_error.append(probability_3)
    return testing_data_error

'''
Case: 1 prior_3 = 0.5 and  prior_7 = 0.5
'''

print(f"The Probability of error of traing data when prior probabilities are 0.5 and 0.5 {np.mean(probability_of_error_traing_data(0.5,0.5))}")
print(f"The Probability of error of testing data when prior probabilities are 0.5 and 0.5 {np.mean(probability_of_error_testing_data(0.5,0.5))}")

'''
case: 2 prior_3 = 0.3 and prior_7 = 0.7
'''
print(f"The Probability of error of traing data when prior probabilities are 0.3 and 0.7 {np.mean(probability_of_error_traing_data(0.3,0.7))}")
print(f"The Probability of error of testing data when prior probabilities are 0.3 and 0.7 {np.mean(probability_of_error_testing_data(0.3,0.7))}")

        
















