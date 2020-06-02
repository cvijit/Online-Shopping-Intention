# 1- Viewing the data 
osi = read.csv( "/Users/vijitchekkala/Desktop/DML/ml_dataset/OSI.csv")
head(osi)
str(osi)
summary(osi)
summary(osi$VisitorType)
summary(osi$Month)
osi$Revenue<-as.factor(osi$Revenue) #converting log into factor
osi$Weekend <- as.factor(osi$Weekend)
str(osi)
table(osi$Revenue)
prop.table(table(osi$Revenue))

#feature selection
#install.packages('Boruta')
#install.packages('ranger')
#library(ranger)
#library(Boruta)
#f <- Boruta(Revenue ~.,data = osi,doTrace=2)# running this will take 10 mins
#print(feat)
#plotImpHistory(feat)

#EDA
#using packages for correlation
library(ggplot2)
library(dplyr)
#
table(osi$Weekend)
counts1 <- table(osi$Weekend)
barplot(counts1, main="Revenue generated on Weekend",
        xlab="Weekend",col = c("blue","skyblue"))

#
table(osi$SpecialDay)
counts <- table(osi$SpecialDay)
barplot(counts, main="Revenue generated on Special Day",
        xlab="Special Day",col=c("blue","blue","blue","skyblue","skyblue","skyblue"))


#class imbalance problem
imbalance <- prop.table(table(osi$Revenue))
print(imbalance)
barplot(imbalance,col=c('darkblue','skyblue'),ylim=c(0,1),main='Imbalance result') #class imbalance can be seen here
library(caret)
confusionMatrix(predict(p1,validate), validate$Revenue,positive = 'True')

#splitting the data
pd <- sample(2,nrow(osi),replace = TRUE,prob = c(0.8,0.2))
train <- osi[pd==1,]
test <- osi[pd==2,]


#decision tree with party package
#install.packages('party')
library(party)
tree1 <- ctree(Revenue ~.,data=train,controls = ctree_control(mincriterion=0.99,minsplit=500))
tree1
plot(tree1)#interpretion of the tree

#predict
predict(tree1,test,type='prob')
#or
predict(tree1,test)

#decision tree with r part
library(rpart)
library(rpart.plot)
tree21 <- rpart(Revenue~., data= train,method='class')
printcp(tree21) #cross validation results #values to be included in for prediction
plot(tree21,uniform = TRUE,main='Revenue')
text(tree21, use.n=TRUE, all=TRUE)
#
tree2 <- rpart(Revenue ~.,train) #variables selected from cross validation is sleected automatically
#plot all ghe below tree diagrams and interpret them from the youtube
rpart.plot(tree2,box.palette=c("skyblue", "royalblue")) 
rpart.plot(tree2,extra = 1,box.palette=c("skyblue", "royalblue")) 
rpart.plot(tree2,extra = 2,box.palette=c("skyblue", "royalblue")) 
rpart.plot(tree2,extra = 3,box.palette=c("skyblue", "royalblue")) 
rpart.plot(tree2,extra = 4,box.palette=c("skyblue", "royalblue")) 
#interpret any two of the following tree diagram


#prediction
predict(tree2,test)

#misclassification error on train data
tab <- table(predict(tree1),train$Revenue)
print(tab)
error1 <- 1-sum(diag(tab))/sum(tab)
error1
#error is about 10 percent which is good not 100 percent

#misclassification error on validate data
testPred <- predict(tree1,newdata = test)
tab1 <- table(testPred,test$Revenue)
print(tab1)
error2 <- 1-sum(diag(tab1))/sum(tab1)
error2
#again 10.8 percent error



#handling classification imbalance problem
#class imbalance problem
imbalance <- prop.table(table(osi$Revenue))
print(imbalance)
barplot(imbalance,col=c('darkblue','skyblue'),ylim=c(0,1),main='Imbalance result') #class imbalance can be seen here
library(caret)
confusionMatrix(predict(p1,validate), validate$Revenue,positive = 'True')

#data for developing predictive model
table(osi$Revenue)
prop.table(table(osi$Revenue))
summary(train)
#prediction model - random forest
library(randomForest)
rftrain <- randomForest(Revenue ~., data = train) 
rftrain
#predictive model evaluation with test data
library(caret)
library(e1071)
confusionMatrix(predict(rftrain,test),test$Revenue,positive = 'TRUE')
#here no information rate is less than accuracy which means that this is a good model
#sensitivity is correctly prediction of TRUE
##specificity is correctly prediction of FALSE
#here the specificity value is much greater than sensitivity

#oversampling for better sensitivity
#install.packages('ROSE')
table(train$Revenue)
prop.table(table(train$Revenue))
library(ROSE)
library(rpart)
imb <- rpart(Revenue ~ ., data =train)
pred.imb <- predict(imb, newdata = test)
head(pred.imb)
head(test$Revenue)
accuracy.meas(test$Revenue, pred.imb[,2])
roc.curve(test$Revenue, pred.imb[,2], plotit = T) #explain ROC curve

data_balanced_over <- ovun.sample(Revenue ~ ., data = train, method = "over",N = 16758)$data
table(data_balanced_over$Revenue)
str(data_balanced_over)


data_balanced_under <- ovun.sample(Revenue ~ ., data = train, method = "under",N = 3054)$data
str(data_balanced_under)
table(data_balanced_under$Revenue)

data_balanced_both <- ovun.sample(Revenue ~ ., data = train, method = "both",p=0.5,N = 9866)$data
str(data_balanced_both)
table(data_balanced_both$Revenue)

data.rose <- ROSE(Revenue~ ., data = train)$data
table(data.rose$Revenue)
#
tree.over <- rpart(Revenue ~ ., data = data_balanced_over)

tree.under <- rpart(Revenue ~ ., data = data_balanced_under)

tree.both <- rpart(Revenue ~ ., data = data_balanced_both)

tree.rose <- rpart(Revenue ~ ., data = data.rose)
#


pred.tree.over <- predict(tree.over, newdata = test)

pred.tree.under <- predict(tree.under, newdata = test)

pred.tree.both <- predict(tree.both, newdata = test)

pred.tree.rose <- predict(tree.rose, newdata = test)
#
roc.curve(test$Revenue, pred.tree.over[,2], plotit = T)
roc.curve(test$Revenue, pred.tree.under[,2], plotit = T)
roc.curve(test$Revenue, pred.tree.both[,2], plotit = T)
roc.curve(test$Revenue, pred.tree.rose[,2], plotit = T)





#random forest imbalanced data


oversampling <- ovun.sample(Revenue~. ,data = train, method='over', N=16612)$data
table(oversampling$Revenue)
summary(oversampling)

rfover <- randomForest(Revenue ~., data=oversampling) 
rfover
confusionMatrix(predict(rfover,test),test$Revenue,positive = 'TRUE') #no


undersampling <- ovun.sample(Revenue~. ,data = train, method='under', N=3120)$data
table(undersampling$Revenue)
summary(undersampling)

rfunder <- randomForest(Revenue ~., data=undersampling) 
rfunder
confusionMatrix(predict(rfunder,test),test$Revenue,positive = 'TRUE')#better
#undersampling is better than oversampling and both


both<- ovun.sample(Revenue~. ,data = train, method='both', p= 0.5,N=9866)$data
table(both$Revenue)
summary(both)

rfboth <- randomForest(Revenue ~., data=both) 
confusionMatrix(predict(rfboth,test),test$Revenue,positive = 'TRUE')













