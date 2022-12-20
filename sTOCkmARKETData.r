#Loading the Stock Market Data.
# Target variable: Direction.
# Method: Classification.
head(Smarket)
sm <- Smarket

# A data.frame: 6 × 9 	
#   Year	Lag1	Lag2	Lag3	Lag4	Lag5	Volume	Today	Direction
# 	<dbl>	<dbl>	<dbl>	<dbl>	<dbl>	<dbl>	<dbl>	<dbl>	<fct>
# 1	2001	0.381	-0.192	-2.624	-1.055	5.010	1.1913	0.959	Up
# 2	2001	0.959	0.381	-0.192	-2.624	-1.055	1.2965	1.032	Up
# 3	2001	1.032	0.959	0.381	-0.192	-2.624	1.4112	-0.623	Down
# 4	2001	-0.623	1.032	0.959	0.381	-0.192	1.2760	0.614	Up
# 5	2001	0.614	-0.623	1.032	0.959	0.381	1.2057	0.213	Up
# 6	2001	0.213	0.614	-0.623	1.032	0.959	1.3491	1.392	Up 


# Splitting the data into testing and training sets.
rows= sample(nrow(sm),floor(0.80*nrow(sm)))
sm_train = sm[rows,]
sm_test=sm[-rows,]

# Converting the inputs to matrix and outputs to categorical.
train_x =sm_train %>% 
 select(-Direction) %>% 
 scale()
train_y = to_categorical(matrix(as.numeric(sm_train$Direction)-1))
 
test_x = sm_test %>% 
 select(-Direction) %>% 
 scale()
test_y = to_categorical(matrix(as.numeric(sm_test$Direction)-1))

# Building the Model.
model = keras_model_sequential() %>% 
  layer_dense(units=10, activation = "relu", input_shape = ncol(train_x)) %>%
    layer_dropout(rate=0.4) %>%
    layer_dense(units=2, activation = "sigmoid") 
  
# Synopsis of the model.
summary(model)

# Model: "sequential_4"
# ________________________________________________________________________________
#  Layer (type)                       Output Shape                    Param #     
# ================================================================================
#  dense_8 (Dense)                    (None, 10)                      90          
#  dropout_3 (Dropout)                (None, 10)                      0           
#  dense_7 (Dense)                    (None, 2)                       22          
# ================================================================================
# Total params: 112
# Trainable params: 112
# Non-trainable params: 0
# ________________________________________________________________________________

# Assigning additional params to the model.
model %>% compile(optimizer = "rmsprop", 
                  loss = "binary_crossentropy",  
                  metric=c("accuracy"))

# Fitting the model.
model %>%
  fit (train_x , train_y , epochs = 50, batch_size = 50,validation_split = 0.2)

# Predicting new values.
y_nn = model %>% predict(test_x) 

# Evaluating the model.
scores = model %>% evaluate(test_x, test_y)
sc=paste('The Accuracy of the model is',scores[2]*100)
print(sc)

# [1] "The Accuracy of the model is 99.5999991893768"

# Logistic regression on the Stock Market Data.
logi_reg <- nnet::multinom(Direction ~., data = sm_train)

# weights:  10 (9 variable)
# initial  value 693.147181 
# iter  10 value 8.648843
# iter  20 value 5.150940
# iter  30 value 4.099196
# iter  40 value 2.525967
# iter  50 value 1.438979
# iter  60 value 1.432693
# iter  70 value 1.423518
# iter  80 value 1.358396
# iter  90 value 1.335911
# iter 100 value 1.330484
# final  value 1.330484 
# stopped after 100 iterations

# Synopsis of the model.
summary(logi_reg)

# Call:
# nnet::multinom(formula = Direction ~ ., data = sm_train)

# Coefficients:
#                    Values    Std. Err.
# (Intercept) -15.079150385 0.0001580044
# Year          0.001909724 0.0005732934
# Lag1         -0.391493998 0.6318983134
# Lag2          1.535010233 0.6212590660
# Lag3         -0.315999618 1.6609588866
# Lag4          0.403267479 0.5207272545
# Lag5          1.399146771 1.4248018164
# Volume        9.418633153 0.2836383233
# Today       378.472402577 0.0021749244

# Residual Deviance: 2.660969 
# AIC: 20.66097 

# Predicting new values.
log_reg_preds = predict(logi_reg,sm_test)

#Accuracy of the Logistic Regression model.
confusionMatrix(log_reg_preds,sm_test$Direction)

# Confusion Matrix and Statistics

#           Reference
# Prediction Down  Up
#       Down  118   0
#       Up      1 131
                                          
#                Accuracy : 0.996           
#                  95% CI : (0.9779, 0.9999)
#     No Information Rate : 0.524           
#     P-Value [Acc > NIR] : <2e-16          
                                          
#                   Kappa : 0.992           
                                          
#  Mcnemar's Test P-Value : 1               
                                          
#             Sensitivity : 0.9916          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.9924          
#              Prevalence : 0.4760          
#          Detection Rate : 0.4720          
#    Detection Prevalence : 0.4720          
#       Balanced Accuracy : 0.9958          
                                          
#        'Positive' Class : Down 

# Accuracy of the logistic Regression model is 99.6%.

# The accuracies of the both the models are on par and fail to show a significant difference.








