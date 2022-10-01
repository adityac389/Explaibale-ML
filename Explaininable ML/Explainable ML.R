##############################################################################################################################################
## This is an Explainable Machine Learning Project which looks at the factors that affect the price of apartments and an aprtment in Warsaw ##
##############################################################################################################################################

## Install Packages
packages_to_install <- c('DALEX', 'randomForest', 'dplyr', 'localModel', 'DALEXtra')
install.packages(packages_to_install)


## Library
library(DALEX)
library(randomForest)
library(dplyr)


## Data
data("apartments")
data("apartments_test")


## Transform_data
data_transform <- function(data){
  data %>%
    mutate(random_var = runif(dim(data)[1]),
    no.rooms = as.factor(no.rooms),
    floor = as.factor(floor))
}

apartments <- apartments %>% data_transform()
apartments_test <- apartments_test %>% data_transform()

head(apartments)


## Making Models
set.seed(220906)

# Train randomForest
apartments_rf_model <- randomForest::randomForest(m2.price ~., data=apartments)

# Train LinearModel
apartments_lm_model <- lm(m2.price ~., data=apartments)

# predict on test set
predicted_rf <- predict(apartments_rf_model, apartments_test)
predicted_lm <- predict(apartments_lm_model, apartments_test)


## Creating Explainers
explainer_lm <- DALEX::explain(model=apartments_lm_model,
                               data = apartments_test[, 2:7],
                               y = apartments_test$m2.price)

explainer_rf <- DALEX::explain(model=apartments_rf_model,
                               data = apartments_test[, 2:7],
                               y = apartments_test$m2.price)


## Model Performance
mp_lm <- model_performance(explainer_lm)
mp_lm
mp_rf <- model_performance(explainer_rf)
mp_rf


## Structure of Model Performance Objects
str(mp_lm)
str(mp_rf)
plot(mp_lm, mp_rf, geom = 'boxplot') #can use histogram as well


## Model Diagnostics
md_lm <- model_diagnostics(explainer = explainer_lm)
md_rf <- model_diagnostics(explainer_rf)

plot(md_rf, variable = 'y', yvariable = 'y_hat')
plot(md_lm, variable = 'y', yvariable = 'y_hat')

plot(md_rf, variable = 'y', yvariable = 'residuals', smooth = FALSE)
plot(md_lm, variable = 'y', yvariable = 'residuals', smooth = FALSE)


plot(md_lm, md_rf, variable = 'construction.year', yvariable = "residuals", smooth = FALSE)
plot(md_lm, md_rf, variable = 'surface', yvariable = "residuals", smooth = FALSE) #play with different variables


## Variable Importance
fi_rf <- DALEX::model_parts(explainer = explainer_rf, B = 10) # B = 10 (No. of Bins to use)
fi_lm <- DALEX::model_parts(explainer = explainer_lm, B = 10)

plot(fi_rf, fi_lm)




### Global Explainers (ALL APARTMENTS)
## Partial Dependency (Categorical)
pdp_cat_rf <- model_profile(explainer_rf, variables = 'district', type = 'partial')
pdp_cat_lm <- model_profile(explainer_lm, variables = 'district', type = 'partial')

plot(pdp_cat_lm, pdp_cat_rf)


## Partial Dependency (Continuous)
pdp_cont_rf <- model_profile(explainer_rf, variables = 'construction.year', type = 'partial')
pdp_cont_lm <- model_profile(explainer_lm, variables = 'construction.year', type = 'partial')

plot(pdp_cont_lm, pdp_cont_rf)
# Partial Dependencies are dependent on all variables (can make strange assumption sometimes like 30 sq.mt. with 10 rooms)


## Partial Dependency (Accumulated Local Efefct)
ale_rf <-  model_profile(explainer_rf, variables = 'surface', type = 'accumulated')
ale_lm <-  model_profile(explainer_lm, variables = 'surface', type = 'accumulated')

plot(ale_lm, ale_rf)




### Local Explainers (SINGLE APARTMENT: APARTMENT = 5) (Understanding the value of an instance when compared to Global data)
## Break Down Plots
bd_rf <- predict_parts(explainer_rf, new_observation = apartments_test[5,])
bd_lm <- predict_parts(explainer_lm, new_observation = apartments_test[5,])

plot(bd_rf)
plot(bd_lm)


## Shapley Values (Credit Rejection, House Value)
shap_rf <- predict_parts(explainer_rf,
                         new_observation = apartments_test[5,],
                         type = 'shap',
                         N = 50 ,
                         B = 50)
                        #N = 50 (No. of Observation)(How many global values to compare to)

shap_lm <- predict_parts(explainer_lm,
                         new_observation = apartments_test[5,],
                         type = 'shap',
                         N = 50 ,
                         B = 50)

plot(shap_rf)
plot(shap_lm)




### LIME (Used to explain the behavior of AI/ML/Neural Network at a Local Point)


## Library
library(DALEXtra)
library(randomForest)
#library(lime)
library(localModel)


## Making Models
set.seed(06092022)

model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

# Lime Model
lime_apt5 <- predict_surrogate(explainer_rf,
                               predict_model.dalex_explainer,
                               model_type.dalex_explainer,
                               new_observation = apartments_test[5,],
                               n_features = 3,
                               n_permutations = 1000,
                               type = 'lime',
                               seed = 123)
                              #Set seed to get same results after changing args and reverting back to same args
plot(lime_apt5)

# Local Model
local_model_5 <- DALEXtra::predict_surrogate_local_model(explainer_rf,
                                             new_observation = apartments_test[5, ],
                                             nb_permutations = 1000)

plot(local_model_5)
####################
##       END      ##
####################