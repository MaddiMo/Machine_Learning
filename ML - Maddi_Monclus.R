
# Load the dataset
library(dplyr)
df = read.csv("winequality-red.csv", sep=",")

# Create report
library(DataExplorer)
create_report(df)
    # 1. There is no NAs in the dataset
    # 2. All the data from this dataset is numerical (therefore, no need to use dummies the categorical data later on)

introduce(df)
plot_intro(df)
plot_missing(df)
summary(df)

# A EDA Report shows more information than the previous report completed
library(SmartEDA)
ExpReport(df,
          op_file="red_wine.html")

# Check the type of data of the dataframe
sapply(df,class)

        # As there are no categorical variables, there is no need to be change separate the numerical and non-numerical data,
        # and the non-numerical data changed to dummies (0 and/or 1). Otherwise, the following code would be used,
        # a new dataframe would be created and added to the only-numerical dataframe once this has been cleaned. 
        
        # df_No_numerica -> the dataframe with no numerical data
        # library(fastDummies)
        # dummies = fastDummies::dummy_columns(df_NO_numerica, 
        #                                      remove_first_dummy = T,
        #                                      remove_selected_columns = T)

# Check for total number of NAs
total = sum(is.na(df)) # it seems there are none

# Remove the variable "Alcohol" as it will be the target variable,
# as well as "quality" as it is not useful for my study

all = df #  a dataframe will all the original data, just in case it is needed later
# df$alcohol = NULL
df$quality = NULL

# Find outliers  ----------------

    # a) function to find outliers
FindOutliers = function(df) {
  lowerq = quantile(df, 0.25)   
  upperq = quantile(df, 0.75)
  iqr = upperq - lowerq   #rango intercuartílico (IQR)
  extreme.upper = (iqr * 3) + upperq
  extreme.lower = lowerq - (iqr * 3)
  result = df > extreme.upper | df < extreme.lower
  return(result)
} 

  # b) create a new dataframe with bolean data to find whether there is any outlier in the dataset

i = 1
cont = as.data.frame(i:nrow(df))
while (i <= ncol(df)) {
  temp_2 = FindOutliers(df[[i]]) # a logical vector vector
  # this condition, uses a "for" within
  cont = cbind.data.frame(cont,temp_2) # join in columns level
  i=i+1
}
cont[[1]] = NULL # first column was only used as index, therefore, it needs to be removed

# It seems there are no outliers in the dataset. 
# However, it would have been the case that there are outliers, with the following code, 
# a new dataframe would be created without the outliers

j = 1
new = as.data.frame(df[1,]) # it is only instrumental, to create the frame of the dataframe
while (j <= nrow(cont)) {
  if ((rowSums(cont)[j] <= ncol(cont/2)) == T) {
    new = rbind.data.frame(new,df[j,])  
  }
  j = j+1
}
# remove first row as it is duplicated
new = new[-1,] 

# visual representation of a dataframe which shows FALSE when the dataframe and the new dataframe without the outliers
comparation = as.data.frame(df == new) 
# it shows in the console whether and where the FALSE occurs in the comparation dataframe (hence outlier)
which(comparation == FALSE, arr.ind = TRUE)

# There is no need to Normalise the dataset as all the data is quite leveled and small numbers.
# Hence, no normalising will not affect on the analysis 


# CORRELATION  ---------

# The correlation between the predictor variables will be analysed and the 
# VIF (variance Inflaction Factor) for each predictor variable and when a high probability is met. 
# As variability is needed, data variables with high correlation will be removed as it does not 
# provide great information 

library(caret)
library(lattice)

filter = (names(df) %in% c("alcohol"))==F

# Correlation Matrix
cor_matrix = cor(df[,filter])

# high correlation between variables (>= 0.85) 
high_correlation = caret::findCorrelation(cor_matrix, cutoff = 0.85)
    # it seems there are not. Otherwise, the following code would have been used to remove them
df_clean=df[, -high_correlation]


# NULL VARIANCE  ------------

# Columns with a low or null (constant) variance are not useful as for complex models 
# it will overtrain it and add computing 
library(caret)

ZeroVar_table = nearZeroVar(df,allowParallel = T,saveMetrics = T)
filter2=row.names(ZeroVar_table[ZeroVar_table$zeroVar==FALSE & ZeroVar_table$nzv==FALSE,])
df_complete=df[, filter2]



# ----- MODELS -----

set.seed(123)
datos = df_complete

# change name of "alcohol variable" to "y"
datos$y = datos$alcohol
datos$alcohol = NULL


    # 1: LINEAL REGRESSION

# save the prediction variable 
datos_y = as.data.frame(datos$y)

# divide the data 5 times 
index = createMultiFolds(datos$y, k = 5, times = 1) # for cross-validation
acierto=c()
for (i in 1:length(index)){
  datostrain = datos[ index[[i]],]
  datostst = datos[-index[[i]],]
  regresion = lm(y~., data=datostrain)
  prediccion=predict(regresion,datostst) 
  resultado=1-(sum((datostst[,'y']-prediccion)^2)/sum((datostst[,'y']-mean(datostst[,'y']))^2))
  acierto = rbind(acierto,c(resultado))
}
mean(acierto)
    # [1] 0.6622827
regresion = lm(y~., data=datos) # predict "alcohol" taking into account all the variables
summary(regresion)
    # Residual standard error: 0.614 on 1588 degrees of freedom
    # Multiple R-squared:  0.6701,	Adjusted R-squared:  0.668 
    # F-statistic: 322.5 on 10 and 1588 DF,  p-value: < 2.2e-16
acierto
    # [1,] 0.6084170
    # [2,] 0.6949381
    # [3,] 0.6624660
    # [4,] 0.6572459
    # [5,] 0.6883463



    # 2. DECISION TREE

aciertoarbol=c()
index = createMultiFolds(datos$y, k = 5, times = 1) # for cross-validation
for (i in 1:length(index)){
  datostrain = datos[index[[i]],]
  datostst = datos[-index[[i]],]
  arbol = rpart::rpart(y ~ ., data = datostrain, maxdepth = 4)
  prediccion = predict(arbol,datostst)
  resultado = 1-(sum((datostst$y-prediccion)^2)/
                   sum((datostst$y-mean(datostst$y))^2))
  aciertoarbol = rbind(aciertoarbol,c(resultado))
}

# Graficamos 
rpart.plot::prp(arbol, cex=.4,main="Arbol")
rpart.plot::rpart.plot(arbol,cex=0.75)
aciertoarbol
    # [1,] 0.5170717
    # [2,] 0.3972798
    # [3,] 0.5347885
    # [4,] 0.4872346
    # [5,] 0.3945335
mean(aciertoarbol)
    # [1] 0.4661816
  
# Nodos terminales
length(unique(prediccion))
length(unique(datostst$strength))

# Error  
ver = cbind.data.frame(datostst$y,prediccion)
ver$Error = ver$`datostst$y`-ver$prediccion
mean(abs(ver$Error))
quantile(ver$Error,probs = c(0.05,0.95))
hist(ver$Error,main = 'Histograma del error', xlab = 'Error' )
plot(ver$`datostst$y`,ver$prediccion)
# Graficamos 
rpart.plot::prp(arbol, cex=.4,main="Arbol")
rpart.plot::rpart.plot(arbol,cex=0.75)


    # 3: RANDOM FOREST

library(randomForest)
aciertoRF=c()
index = createMultiFolds(datos$y, k = 5, times = 1) # for cross-validation
aciertoRFtrain=c()
for (i in 1:length(index)){
  datostrain = datos[index[[i]],]
  datostst = datos[-index[[i]],]
  modeloRF = randomForest::randomForest(y ~ ., data=datostrain, ntree=1000, mtry=3)
  predicciontrain=predict(modeloRF,datostrain)
  prediccion=predict(modeloRF,datostst)
  resultado=1-(sum((datostst$y-prediccion)^2)/sum((datostst$y-mean(datostst$y))^2))
  resultadotrain=1-(sum((datostrain$y-predicciontrain)^2)/sum((datostrain$y-mean(datostrain$y))^2))
  aciertoRF = rbind(aciertoRF,c(resultado))
  aciertoRFtrain = rbind(aciertoRFtrain,c(resultadotrain))
}

ver = cbind.data.frame(datostst$y,prediccion)
ver$Error = ver$`datostst$y`-ver$prediccion
mean(abs(ver$Error))
quantile(ver$Error,probs = c(0.05,0.95))
aciertoRF
    # [1,] 0.7121870
    # [2,] 0.7440170
    # [3,] 0.7674718
    # [4,] 0.7636052
    # [5,] 0.7518755
mean(aciertoRF)
    # [1] 0.7478313
mean(aciertoRFtrain)
    # [1] 0.9532123
aciertototal=cbind(aciertoRF,aciertoRFtrain)
aciertototal
importanciarf=as.data.frame(importance(modeloRF))


    # 4. XGBoost

library(xgboost)

set.seed(123)
aciertoXGBoost = c()
y = which(names(datos) == "y")
index = createMultiFolds(datos$y, k = 5, times = 1) # for cross-validation
for (i in 1:length(index)){
  datostrain = datos[ index[[i]],]
  datostst = datos[-index[[i]],]
  train_y=datostrain$y
  train_x=datostrain[,-y]
  test_x=datostst[,-y]
  test_y=datostst$y
  
  dtrain = xgb.DMatrix(as.matrix(train_x),label = train_y)
  dtest = xgb.DMatrix(as.matrix(test_x))
  
  xgb_params = list(colsample_bytree = 1, # number of variables in each tree 
                     subsample = 1, 
                     booster = "gbtree",
                     max_depth = 9,
                     min_child_weight=1.5,
                     reg_alpha=0.8,
                     reg_lambda=0.6,
                     set.seed=123,
                     eta = 0.03, 
                     eval_metric = "rmse", 
                     objective = "reg:squarederror",
                     gamma = 0)
  
  gb_dt = xgb.train(xgb_params,dtrain,nfold = 12,nrounds = 1000)
  prediccionxgb = predict(gb_dt,dtest)
  resultado=1-(sum((datostst[,y]-prediccionxgb)^2)/sum((datostst[,y]-mean(datostst[,y]))^2))
  aciertoXGBoost = rbind(aciertoXGBoost,c(resultado))
}

  # Acierto
aciertoXGBoost
    # [1,] 0.7476258
    # [2,] 0.8139126
    # [3,] 0.8293627
    # [4,] 0.8016442
    # [5,] 0.8163161
mean(aciertoXGBoost)
    # [1] 0.8017723

# Importance of the variables
library(tibble)
imp_matrix = as_tibble(xgb.importance(feature_names = colnames(test_x), model = gb_dt))
imp_matrix
    # Feature                Gain  Cover Frequency
    # 1 density              0.622  0.182     0.114 
    # 2 residual.sugar       0.0852 0.113     0.0961
    # 3 fixed.acidity        0.0568 0.0996    0.0914
    # 4 sulphates            0.0539 0.0943    0.100 
    # 5 total.sulfur.dioxide 0.0420 0.0882    0.102 
    # 6 citric.acid          0.0419 0.0884    0.0938
    # 7 pH                   0.0372 0.0929    0.0982
    # 8 volatile.acidity     0.0266 0.0976    0.113 
    # 9 chlorides            0.0228 0.0821    0.114 
    # 10 free.sulfur.dioxide  0.0114 0.0614    0.0779

  # the matrix shows that the variable which higher impact to take into account is 
  # "density", which explains the behaviour of the data in a 62.2%

# Terminal Nodes 
length(unique(prediccionxgb))
    # 305
length(unique(datostst$y))
    # 49

# Error
ver = cbind.data.frame(datostst$y,prediccionxgb)
ver$Error = ver$`datostst$y`-ver$prediccionxgb
mean(abs(ver$Error))
    # [1] 0.3075905
quantile(ver$Error,probs = c(0.05,0.95))
    # 5%        95% 
    #   -0.6368450  0.8445711 
plot(ver$`datostst$y`,ver$prediccion)



    # RESULT

# From the different models that have been completed, it has been seen that
# the XGBoost is the best one.From the result,it can be said that XGBoost 
# provides a considerable good enough. # However, it does not provide a definitive answer


