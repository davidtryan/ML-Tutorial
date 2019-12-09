#' ---
#' title: An example of practical machine learning using R
#' author: Cheng Juan
#' output: 
#'    html_document:
#'        toc: true
#'        highlight: zenburn
#' ---

#+ echo+FALSE
opts_chunk$set(results='asis')
opts_chunk$set(warning=FALSE)
opts_chunk$set(echo=FALSE)

#' ## Summary
#' 
#' The R language has a rich set of modeling functions for classfication. 'caret' tries to
#' generalize and simplify the model building process by eliminating syntactical differences
#' between models.
#' 
#' The aim of the dataset is to build a prediction model on common incorrect
#' gestures during barbell lifts based on several variables collected by accelerometers.
#' 
#' The steps that we need to take to find an accurate prediction model are as follows:
#' 1) Eliminate redundance features with too many missing values
#' 2) Divide remaining dataset into parts:
#'   * training set  
#'   * validation set  
#'   * test set
#' 3) Use train set to train four models: a) classification tree, b) random forest
#' c) boosting, d) bagging
#' 4) Measure out of sample accuracy for models using validation set
#' 5) Use out of sample accuracy to select findal model
#' 6) Use final model to analyze test set
#' 
#' NOTE: The random forest will be selected as the final moddel with
#' an overall accuracy of 0.9946 (with out of sample accuracy from validation
#' set). Using the model to analyze the test set, we obtain a test accuracy of 100%.



#' ## Background
#' 
#' ## Background of the data
#' Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to 
#' collect a large amount of data about personal activity relatively inexpensively. 
#' In this project, we use data from accelerometers on the belt, forearm, arm, and 
#' dumbell of 6 participants. Six young health participants were asked to perform one 
#' set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different 
#' fashions: exactly according to the specification (Class A), throwing the elbows 
#' to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the 
#' dumbbell only halfway (Class D) and throwing the hips to the front (Class E).
#' 
#' ## Technical Background

library(caret)

build <- read.csv("Documents/DataScience/Tutorial/PracticalMachineLearning/WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv")
dim(build)
test_ind <- sample(1:nrow(build), nrow(build)*0.001)
test <- build[test_ind,]
build <- build[-test_ind,]

dim(build)
dim(test)
str(build)
str(test)

build[,6:158] <- sapply(build[,6:158], as.numeric)
test[,6:158] <- sapply(test[,6:158], as.numeric)

build <- build[,7:159]
test <- test[,7:159]

#' ## Search for missings
install.packages('Amelia')
library(Amelia)

missmap(test, main = "Missingness Map Test")

# since test set only contains 20 observations. 
# remove features that contains NAs in test set
nas <- is.na(apply(test[1:152],2,sum))

test <- test[,c(as.numeric(which(!nas)),153)]
dim(test)
build <- build[,c(as.numeric(which(!nas)),153)]
dim(build)

# create validation data set using Train 
inTrain <- createDataPartition(y=build$classe, p=0.7, list=FALSE)
train <- build[inTrain,]
val <- build[-inTrain,]
rm(inTrain,nas,build)

sumtable <- data.frame(c('training','validation','test'), c(nrow(train),nrow(val),nrow(test)), c(ncol(train),ncol(val),ncol(test)))
colnames(sumtable) <- c('Dataset','# of observations','# of features')
library(knitr)
knitr::kable(sumtable)


#' ## Classification Model
#' 
#' #### Classification Tree
#' 
#' In the first test, we use a regression tree with the method rpart.
library(rattle)
library(rpart.plot)
library(rpart)
#  regression tree model
set.seed(123)
Mod0 <- train(classe ~ .,data=train, method="rpart")
save(Mod0,file="Documents/DataScience/Tutorial/PracticalMachineLearning/Mod0.RData")

load("Mod0.RData")
fancyRpartPlot(Mod0$finalModel)

# out-of-sample errors of regression tree model using validation dataset 
pred0 <- predict(Mod0, val[,which(colnames(val)!="classe")])
cm0 <- confusionMatrix(pred0, val$classe)
cm0$table
kable(cm0$table)



#' ## Random Forest
#' 
set.seed(123)

# random forest model
system.time(Mod1 <- train(classe ~ ., method = "rf", 
               data = train, importance = T, 
               trControl = trainControl(method = "cv", number = 3)))
save(Mod1,file="Mod1.RData")

load("Mod1.RData")
Mod1$finalModel

vi <- varImp(Mod1)
vi$importance[1:10,]

# out-of-sample errors of random forest model using validation dataset 
pred1 <- predict(Mod1, val[,which(colnames(val)!="classe")])
cm1 <- confusionMatrix(pred1, val$classe)


# plot roc curves
library(pROC)
pred1.prob <- predict(Mod1, val, type="prob")
pred1.prob$
roc1 <-  roc(val$total_accel_belt, pred1.prob$E)
plot(roc1, print.thres="best", print.thres.best.method="closest.topleft")
coord1 <- coords(roc1, "best", best.method="closest.topleft",
                          ret=c("threshold", "accuracy"))
coord1


# summary of final model
Mod1$finalModel
plot(Mod1)