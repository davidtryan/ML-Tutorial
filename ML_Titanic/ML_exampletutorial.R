# http://bigcomputing.blogspot.com/2014/10/an-example-of-using-random-forest-in.html
# http://will-stanton.com/machine-learning-with-r-an-irresponsibly-fast-tutorial/
  
### Install relevant packages
install.packages('caret', dependencies=T)
install.packages('randomForest')

library(caret)
library(randomForest)

### Set working directory
setwd("Documents/DataScience/Tutorial/Titanic/")

### Load data
trainSet <- read.table("train.csv", sep=",", header=T)
testSet <- read.table("test.csv", sep=",", header=T)

### Explore data
dim(trainSet)
head(trainSet)
dim(testSet)
head(testSet)
colnames(trainSet)[!colnames(trainSet) %in% colnames(testSet)]
#NOTE: We need to be able to make predictions about SURVIVAL on the unlabeled (test) data set



### Testing for useful variables
# Here we must pick the most meaningful/useful variables

#### Crosstabs for categorical variables
# Crosstabs show the interactions between two variables in a very easy to read way.
str(trainSet)
# sex, sibsp, parch, embarked
table(trainSet[,c('Survived','Pclass')])
plot(table(trainSet[,c('Survived','Pclass')]))   #YES

table(trainSet[,c('Survived','Sex')])
plot(table(trainSet[,c('Survived','Sex')]))  #YES

table(trainSet[,c('Survived','SibSp')])
plot(table(trainSet[,c('Survived','SibSp')]))  #NO (tutorial says YES)

table(trainSet[,c('Survived','Parch')])
plot(table(trainSet[,c('Survived','Parch')]))   #NO (tutorial says YES)

table(trainSet[,c('Survived','Embarked')])
plot(table(trainSet[,c('Survived','Embarked')]))   #NO (tutorial says YES)

#### Plots for continuous variables
install.packages('fields')
library(fields)
# Age and Fare are the only really continuous variables in the data set
bplot.xy(trainSet$Survived, trainSet$Age)   #boxplots same - NO
summary(trainSet$Age)   #lots of NAs - NO

bplot.xy(as.integer(trainSet$Survived), trainSet$Fare)   #YES
summary(trainSet$Fare)   #lots of NAs - NO


### Training a Model
# convert Survived to a Factor data type, so that caret builds a classification instead of a regression model
# Convert Survived to Factor
trainSet$Survived <- as.factor(trainSet$Survived)
# Set a random seed (so you will get repeated results)
set.seed(42)
trainSet_ind <- sample(nrow(trainSet), nrow(trainSet)*0.8)
validSet_ind <- (1:nrow(trainSet))[!(1:nrow(trainSet)) %in% trainSet_ind]
#Also could use split() or createDataPartition()
validSet <- trainSet[validSet_ind,]
trainSet <- trainSet[trainSet_ind,]

# Train the model using a random forest algorithm (classify survivors vs non-survivors based on criteria)
model <- train(Survived ~ Pclass + Sex + SibSp +
                 Embarked + Parch + Fare,
               data=trainSet, method='rf',
               trControl=trainControl(method='cv', number=5))
print(model)

varImp(model)
varImpPlot(model, type=1)



### Validate the model
validSet$Survived_P <- predict(model, newdata=validSet[,which(colnames(validSet)!="Survived")])

head(validSet)
# http://www.inside-r.org/node/86995
confusionMatrix(validSet$Survived_P, validSet$Survived)



### Testing the model

testSet$Survived <- predict(model, newdata=testSet)

summary(testSet)
testSet$Fare <- ifelse(is.na(testSet$Fare), mean(testSet$Fare, na.rm=T), testSet$Fare)

testSet$Survived <- predict(model, newdata=testSet)



### Improving the model

# Try including different variables in the model: leave some out or add some in
# Try combining variables into more useful variables: sometimes you can multiply or add variables together, or concatenate different categorical variables together
# Try transforming the existing variables in clever ways: maybe turn a numerical variable into a categorical variable based on different ranges (e.g. 0-10, 10-90, 90-100)
# Try a different algorithm: maybe neural networks, logistic regression or gradient boosting machines work better. Better yet, train a few different types of models and combine the results by averaging the probabilities (this is called ensembling)






