Jawbone Up Wearer Performance Prediction
========================================================
-- Course Project for Practical Machine Learning



Project Description:

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

This human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.

In this work we first define quality of execution and investigate three aspects that pertain to qualitative activity recognition: the problem of specifying correct execution, the automatic and robust detection of execution mistakes, and how to provide feedback on the quality of execution to the user. We tried out an on-body sensing approach, but also an "ambient sensing approach" (by using Microsoft Kinect).

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

========================================================
Project Approaches:

First is to import the datasets, and set the dependent variable "classe" aside. 


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.1
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pml.training <- read.csv("C:/Users/BenBen/Documents/Google Drive/coursera-Data Science Specialization/08_PracticalMachineLearning/project/pml-training.csv", na.strings = c("NA", ""))
pml.testing <- read.csv("C:/Users/BenBen/Documents/Google Drive/coursera-Data Science Specialization/08_PracticalMachineLearning/project/pml-testing.csv", na.strings = c("NA", ""))
classe=pml.training$classe
```

The basic preprocessing


```r
### basic
#1 remove irrelevant
rIndex <- grep("X|user_name|cvtd_timestamp", names(pml.training))
pml.training <- pml.training[, -rIndex]
#2 remove zero var
nzv <- nearZeroVar(pml.training)
pml.training <- pml.training[, -nzv]
#3 remove na
NAs <- apply(pml.training, 2, function(x) {
  sum(is.na(x))
})
pml.training <- pml.training[, which(NAs == 0)]
pml.training0=pml.training
#4 factor to dummy **don't need
```

The advanced preprocessing (which is not used as they don't improve the prediction accuracy)

```r
### extra
#1 impute
preObj <- preProcess(pml.training[,-56],method="medianImpute")
pml.training<- predict(preObj,pml.training[,-56])
pml.training1=cbind(pml.training,classe) #best
#2 standardize
stdize=preProcess(pml.training[,-56],method=c("center","scale"))
pml.training=predict(stdize,pml.training[,-56])
pml.training2=cbind(pml.training,classe)
## check corr
M <- abs(cor(pml.training))
diag(M) <- 0
which(M > 0.8,arr.ind=T)
```

```
##                  row col
## yaw_belt           6   4
## total_accel_belt   7   4
## accel_belt_y      12   4
## accel_belt_z      13   4
## accel_belt_x      11   5
## magnet_belt_x     14   5
## roll_belt          4   6
## roll_belt          4   7
## accel_belt_y      12   7
## accel_belt_z      13   7
## pitch_belt         5  11
## magnet_belt_x     14  11
## roll_belt          4  12
## total_accel_belt   7  12
## accel_belt_z      13  12
## roll_belt          4  13
## total_accel_belt   7  13
## accel_belt_y      12  13
## pitch_belt         5  14
## accel_belt_x      11  14
## gyros_arm_y       22  21
## gyros_arm_x       21  22
## magnet_arm_x      27  24
## accel_arm_x       24  27
## magnet_arm_z      29  28
## magnet_arm_y      28  29
## accel_dumbbell_x  37  31
## accel_dumbbell_z  39  32
## gyros_dumbbell_z  36  34
## gyros_forearm_z   49  34
## gyros_dumbbell_x  34  36
## gyros_forearm_z   49  36
## pitch_dumbbell    31  37
## yaw_dumbbell      32  39
## gyros_forearm_z   49  48
## gyros_dumbbell_x  34  49
## gyros_dumbbell_z  36  49
## gyros_forearm_y   48  49
```

```r
which(M > 0.9,arr.ind=T)
```

```
##                  row col
## total_accel_belt   7   4
## accel_belt_y      12   4
## accel_belt_z      13   4
## accel_belt_x      11   5
## roll_belt          4   7
## accel_belt_y      12   7
## accel_belt_z      13   7
## pitch_belt         5  11
## roll_belt          4  12
## total_accel_belt   7  12
## accel_belt_z      13  12
## roll_belt          4  13
## total_accel_belt   7  13
## accel_belt_y      12  13
## gyros_arm_y       22  21
## gyros_arm_x       21  22
## gyros_dumbbell_z  36  34
## gyros_forearm_z   49  34
## gyros_dumbbell_x  34  36
## gyros_forearm_z   49  36
## gyros_dumbbell_x  34  49
## gyros_dumbbell_z  36  49
```

```r
#3 pca
pca<- preProcess(pml.training,method="pca",thresh=0.9)
pml.training_=predict(pca,pml.training)
pml.training3=cbind(pml.training_,classe)
```


Here is the main model part. I used the two best performing models RF and GBM.
```
### model
set.seed(12345)
inTrain = createDataPartition(pml.training0$classe,p=3/4,list = FALSE)
pml.train<- pml.training0[inTrain, ]  
pml.test <- pml.training0[-inTrain,]  
modFit1<-train(pml.train$classe~.,data=pml.train,method="rf")
modFit2<-train(pml.train$classe~.,data=pml.train,method="gbm")
```
