import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd

# #****************************************************************
# #              Preprocessing                                    *
# #****************************************************************

#Loading the train_data1 and train_data2
train_data1= scipy.io.loadmat('/Users/raji/Documents/2nd sem/FSL/Project/training_data_0.mat')
train_data2= scipy.io.loadmat('/Users/raji/Documents/2nd sem/FSL/Project/training_data_1.mat')

#Converting the train_data1 to 784 dimensional vector
images_train_data_0 = []
for i in range(len(train_data1['nim0'][0][0])):
    a= np.array(train_data1.get('nim0')[0:28,0:28,i])
    images_train_data_0.append(a.flatten())

#Converting the train_data2 to 784 dimensional vector
images_train_data_1 = []
for i in range(len(train_data2['nim1'][0][0])):
    a= np.array(train_data2.get('nim1')[0:28,0:28,i])
    images_train_data_1.append(a.flatten())

#Concatenating both train_data1 images and train_data2 images and compute the total length
res= np.concatenate((images_train_data_0, images_train_data_1))
len_res_split = len(images_train_data_0)
total_length_train = len(res)

#Loading the test_data1 and test_data2
test_data1=scipy.io.loadmat('/Users/raji/Documents/2nd sem/FSL/Project/testing_data_0.mat')
test_data2=scipy.io.loadmat('/Users/raji/Documents/2nd sem/FSL/Project/testing_data_1.mat')

#Converting the test_data1 to 784 dimensional vector
images_test_data_0 = []
for i in range(len(test_data1['nim0'][0][0])):
    a= np.array(test_data1.get('nim0')[0:28,0:28,i])
    images_test_data_0.append(a.flatten())

#Converting the test_data2 to 784 dimensional vector
images_test_data_1 = []
for i in range(len(test_data2['nim1'][0][0])):
    a= np.array(test_data2.get('nim1')[0:28,0:28,i])
    images_test_data_1.append(a.flatten())

#Concatenating both test_data1 images and test_data2 images and compute the total length
res_test= np.concatenate((images_test_data_0, images_test_data_1))
len_res_split_test = len(images_test_data_0)
total_length_test = len(res_test)
#****************************************************************
#      TASK-1: FEATURE - NORMALIZATION                          *
#****************************************************************

print(" Computing Feature Normalization ")
def normalize(imgarray):
    arr = np.asarray(imgarray)
    xmatrix = np.zeros((len(imgarray),len(imgarray[0])))
    mean_i = np.zeros(len(imgarray[0]))
    sd_i = np.zeros(len(imgarray[0]))
    for i in range(len(imgarray[0])):  # 784
        mean_i[i] = np.mean(arr[:, i])
        sd_i[i] = np.std(arr[:, i], dtype=np.float64)
    for i in range(len(arr[0])):  # 784
        for j in range(len(arr)):  # 5923
            xmatrix[j][i] = (arr[j][i] - mean_i[i]) / sd_i[i]
    return xmatrix

df_norm = normalize(res)
df_test_norm = normalize(res_test)

# #****************************************************************
# #                  TASK-2: PCA                                  *
# #****************************************************************
print(" Computing the PCA ")

def calculatePCA(normalized_matrix, title):
    covariance_matrix = np.cov(normalized_matrix.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    print("Eigen Vectors of : ",title ," data ")
    print(eigen_vectors)
    print("Eigen Values of : ", title, " data "),
    print(eigen_values)
    variance_explained = []
    for i in eigen_values:
        variance_explained.append((i / sum(eigen_values)) * 100)
    print('Cumulative Variance')
    cumulative_variance_explained = np.cumsum(variance_explained)
    print(cumulative_variance_explained)
    projection_matrix = (eigen_vectors.T[:][:2]).T
    X_pca = normalized_matrix.dot(projection_matrix)
    return X_pca

train_pca = calculatePCA(df_norm,'Train')
test_pca = calculatePCA(df_test_norm, 'Test')
print(train_pca)

# #****************************************************************
# #                 TASK-3: Plotting                             *
# #****************************************************************
print("Performing Dimensionality Reduction and plotting the data")
def dimension_redcution(pca, split, total_length, title):
    for i in range(0,split):
        y1 = plt.scatter(pca[i,0], pca[i,1], c='orange')
    for i in range(split,total_length):
        y2 = plt.scatter(pca[i,0], pca[i,1], c='green')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(handles = [y1, y2],
               labels  = ['Digit 0', 'Digit 1'])
    plt.title(title)
    plt.show()

dimension_redcution(train_pca,len_res_split,total_length_train,"train")
dimension_redcution(test_pca,len_res_split_test,total_length_test,"test")
#
# #****************************************************************
# #                  TASK-4: Plotting                             *
# #****************************************************************
digit_0 = train_pca[:5923, :]
digit_1 = train_pca[5923:12665, :]
test_0 = test_pca[:980, :]
test_1 = test_pca[980:2115, :]


# Mean of Class 0 and Class 1
mu0=(np.mean(digit_0,axis=0))
mu1=(np.mean(digit_1,axis=0))

mu0=mu0[:,None]
mu1=mu1[:,None]

cov0 = np.cov(digit_0, rowvar = False, bias =  False)
cov1 = np.cov(digit_1, rowvar = False, bias =  False)


print("################# Parameters of Digit 0 #####################")
print("Mean vector=", mu0)
print("Covariance Matrix=", cov0)
print("################# Parameters of Digit 1 #####################")
print("Mean vector=", mu1)
print("Covariance Matrix=", cov1)

#*********************************************************************************************#
#              TASK - 5 : Bayesian Decision Theory for optimal classification                 #
#*********************************************************************************************#

def calculating_pdf(mu0, mu1, sigma0, sigma1, x):
    class_conditional_image0 = np.exp(
        -0.5 * (np.matmul(np.matmul((x - mu0).transpose(), np.linalg.inv(sigma0)), (x - mu0)))) / (
                                           np.sqrt(np.linalg.det(sigma0)) * 2 * np.pi)
    class_conditional_image1 = np.exp(
        -0.5 * (np.matmul(np.matmul((x - mu1).transpose(), np.linalg.inv(sigma1)), (x - mu1)))) / (
                                           np.sqrt(np.linalg.det(sigma1)) * 2 * np.pi)

    return class_conditional_image0[0][0] / class_conditional_image1[0][0]

def bayes_error(imgarray, true, mu0, mu1, sigma0, sigma1):
    length = len(imgarray)
    errorcount = 0
    for i in range(length):
        x = imgarray[i][:, None]
        if true == 0:
            if calculating_pdf(mu0, mu1, sigma0, sigma1, x) < 1:
                errorcount += 1
        if true == 1:
            if calculating_pdf(mu0, mu1, sigma0, sigma1, x) > 1:
                errorcount += 1
    return errorcount

# Calculating Testing error
test_error=bayes_error(test_0,0,mu0,mu1,cov0,cov1)+bayes_error(test_1,1,mu0,mu1,cov0,cov1)
training_error=bayes_error(digit_0,0,mu0,mu1,cov0,cov1)+bayes_error(digit_1,1,mu0,mu1,cov0,cov1)
print("training data accuracy=",(1-((training_error)/((len(digit_0)+len(digit_1)))))*100,"%")
print("test data accuracy=",(1-((test_error)/(len(test_0)+len(test_1))))*100,"%")
