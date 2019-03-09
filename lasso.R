###########################

# Lasso Regression Implementation from Scratch
# On Ames Housing Dataset

###########################

# import libraries
if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  "caret", "dplyr", "ggplot2", "glmnet", "robustbase", "robustHD", "xgboost", "moments", "MASS", "plyr"
)
# Remove unimportant variables
vars_remove = c("Street", "Utilities", "Condition_2", "Heating", "Pool_QC", "Misc_Feature", "Low_Qual_Fin_SF", "Three_season_porch", "Pool_Area", "Misc_Val", "Longitude", "Latitude")
#################
train = read.csv("train.csv")[, -83]
test = read.csv("test.csv")
test.PID = test[, 1]
train.y = read.csv("train.csv")[, 83]
read_data = rbind(train, test)
all.PID = read_data[,1]
data1 = read_data[, !(names(read_data) %in% vars_remove)]
data1 = data1[, -1]
# Found one of the values in Garage_Yr_Blt is 2207 which is incorrect
# Change value of garage year from 2207 to 2007 which makes most sense
data1$Garage_Yr_Blt[which(data1$Garage_Yr_Blt == 2207)] = 2007
data1$Year_Built = as.factor(as.integer(data1$Year_Built))
data1$Garage_Yr_Blt[is.na(data1$Garage_Yr_Blt)] = data1$Year_Built[is.na(data1$Garage_Yr_Blt)]
data1$Garage_Yr_Blt = NULL

# FEATURE ENGINEERING
# Create new variable Total_SF --> Decreases RMSE by a small bit
data1$Total_SF = data1$Total_Bsmt_SF + data1$First_Flr_SF + data1$Second_Flr_SF
data1$Bldg_Type[which(data1$Bldg_Type == "TwnhsE")] ="Twnhs"
# data1$Bsmt_SF = data1$BsmtFin_SF_1 + data1$BsmtFin_SF_2 + data1$Bsmt_Unf_SF
data1$Exter_SF =  data1$Wood_Deck_SF + data1$Open_Porch_SF + data1$Screen_Porch

# Create variable Age --> Year_Sold - Year_Built
data1$Age_Remod = data1$Year_Sold - data1$Year_Remod_Add
data1$Year_Built = as.integer(data1$Year_Built)
data1$Age = data1$Year_Sold - data1$Year_Built


# Create variable Bsmt_Bath
data1$Bsmt_Bath =  data1$Bsmt_Full_Bath + 0.5*data1$Bsmt_Half_Bath

# Create variable Bath
data1$Bath =  data1$Full_Bath + 0.5*data1$Half_Bath



quality = c('No_Garage'=0, 'Poor' =1, 'Typical'=2, 'Fair'=3, 'Good'=4, 'Excellent'=5)
data1$Garage_Qual = as.integer(revalue(data1$Garage_Qual, quality))
data1$Garage_Cond = as.integer(revalue(data1$Garage_Cond, quality))
data1$Garage_Score  = data1$Garage_Qual * data1$Garage_Cond
data1$Garage_Interaction = data1$Garage_Qual * data1$Garage_Cars
data1$Avg_Rm_Size = (data1$Gr_Liv_Area) / data1$TotRms_AbvGrd
data1$bath_rm_ratio = data1$Bath / (data1$Bedroom_AbvGr+1)

data1$Alley_Qual = with(data1, interaction(Overall_Qual,  Alley), drop = TRUE )
data1$Pave_Cond = with(data1, interaction(Garage_Qual,  Paved_Drive), drop = TRUE )

data1$Overall = with(data1, interaction(Overall_Qual,  Overall_Cond), drop = TRUE )

data1$Lot_Type = with(data1, interaction(Lot_Shape,  Lot_Config), drop = TRUE )
data1$Build_Neigh = with(data1, interaction(Neighborhood, Bldg_Type), drop = TRUE )

data1$Heat_Elec = with(data1, interaction(Heating_QC, Electrical), drop = TRUE )
data1$Neigh_Zone = with(data1, interaction(Neighborhood, MS_Zoning), drop = TRUE )
data1$Lot_Ratio = data1$Lot_Area / data1$Lot_Frontage

data1$Build_House = with(data1, interaction(House_Style, Bldg_Type), drop = TRUE )
data1$Roof_Exter = with(data1, interaction(Roof_Matl, Exterior_1st), drop = TRUE )

data1$House_Roof = with(data1, interaction(House_Style, Roof_Style), drop = TRUE )
data1$Config_Slope = with(data1, interaction(Lot_Config, Land_Slope), drop = TRUE )

data1$Shape_Fence = with(data1, interaction(Fence, Lot_Shape), drop = TRUE)

data1$Fire_Air = with(data1, interaction(Central_Air, Fireplace_Qu), drop = TRUE )
data1$Neigh_Exter1 = with(data1, interaction(Neighborhood, Exterior_1st), drop = TRUE )

overqual = c('Very_Poor'=1, 'Poor' =2, 'Fair'=3, 'Below_Average'=4, 'Average'=5, 
             'Above_Average'=6, 'Good' = 7, 'Very_Good'=8, 'Excellent'= 9, 'Very_Excellent'=10)
data1$Overall_Q = as.integer(revalue(data1$Overall_Qual, overqual))
data1$MVnr_Exter1 = with(data1, interaction(Mas_Vnr_Type, Exterior_1st), drop = TRUE )
data1$MVnr_Exter2 = with(data1, interaction(Mas_Vnr_Type, Exterior_2nd), drop = TRUE )
data1$House_Neigh = with(data1, interaction(House_Style, Neighborhood), drop = TRUE )


# Factor Gr_Liv_Area
data1$Gr_Liv_Area[data1$Gr_Liv_Area < 1198] = 1
data1$Gr_Liv_Area[data1$Gr_Liv_Area >= 1198 & data1$Gr_Liv_Area < 1610] = 2
data1$Gr_Liv_Area[data1$Gr_Liv_Area >= 1610 & data1$Gr_Liv_Area < 1931] = 3
data1$Gr_Liv_Area[data1$Gr_Liv_Area >= 1931 & data1$Gr_Liv_Area < 2323] = 4
data1$Gr_Liv_Area[data1$Gr_Liv_Area >= 2323 & data1$Gr_Liv_Area < 2655] = 5
data1$Gr_Liv_Area[data1$Gr_Liv_Area >= 2655] = 6
data1$Gr_Liv_Area = as.factor(as.integer(data1$Gr_Liv_Area))



winsor = function(data){
  for (idx in 1:length(names(data))){
    if (is.numeric(data[,idx])){
      data[, idx] = winsorize(data[, idx])
    }
  }
  return(data)
}

# # Winsorize the data (PID column not included)
data2 = winsor(data1)
data2[is.na(data2)] = 0


relevel = function(df, threshold){
  for (idx in 1:ncol(df)){
    if (is.factor(df[,idx])){
      other_cat = names(which(table(df[,idx])/length(df[,idx]) < threshold))
      if(length(other_cat)!=0){
        levels(df[, idx]) = c(levels(df[,idx]), "Other")
        df[, idx][ df[,idx] %in% other_cat] = "Other"
      }
    }
  }
  return(df)
}

data2.1 = relevel(data2, 0.009)
data2.1$Mas_Vnr_Area = data1$Mas_Vnr_Area

# # Create dummy variables to use one-hot encoding
data3 <- data.frame(data2.1, row.names = all.PID)
# data3 <- data.frame(data2, row.names = all.PID)
dmy = dummyVars(" ~ .", data = data3)
df1 = data.frame(predict(dmy, newdata = data3))


one_step_lasso = function(r, x, lam){
  xx = sum(x^2)
  xr = sum(r*x)
  b = (abs(xr) -lam/2)/xx
  b = sign(xr)*ifelse(b>0, b, 0)
  return(b)
}

mylasso = function(X, y, lam, n.iter = 220, standardize  = TRUE)
{
  # X: n-by-p design matrix without the intercept
  # y: n-by-1 response vector
  # lam: lambda value
  # n.iter: number of iterations
  # standardize: if True, center and scale X and y. 
  n = nrow(X)  # Define n
  p = ncol(X)  # Define p
  
  
  # If standardize  = TRUE, center and scale X and Y; record the
  # corresponding means and sd
  mean.x = c()
  se.x = c()
  mean.y = sum(y)/n
  se.y = sd(y)/sqrt(n)
  if(standardize == TRUE){
    y = (y - mean.y)/se.y
    for (col in 1:p){
      column = X[, col]
      if((range(column)[2] == 1)){
        # If it is a dummy variable, don't standardize it
        mean.x = append(mean.x, 0)
        se.x = append(se.x, 1)
      }else{
        mean.x = append(mean.x, mean(column))
        if(sd(column) != 0){
          se.x = append(se.x, sd(column)/sqrt(n))
        }else{
          se.x = append(se.x, 1)
        }
        
        X[, col] = (column - mean.x[col]) /se.x[col]
      }
    }
  }
  l_val = lam*2*n
  
  # Initial values for residual vector and coefficient vector beta
  beta = rep(0, p)
  r = y
  
  for(step in 1:n.iter){
    
    for(j in 1:p){
      
      
      # 1) Update the residual vector to be the one
      # in blue on p37 of [lec_W3_VariableSelection.pdf]. 
      # r <-- current residual + X[, j] * b[j]
      r = r + X[,j] * beta[j]
      
      
      
      # 2) Apply one_step_lasso to update beta_j
      beta[j] = one_step_lasso(r, X[, j], lam) 
      
      
      
      # 3) Update the current residual vector
      # r <-- r - X[, j] * b[j]
      r = r - X[,j] * beta[j]
    }
    
  }
  
  # Scale back b and add intercept b0
  # For b0, check p13 of [lec_W3_VariableSelection.pdf]. 
  if(standardize == TRUE){
    for (col in 1:p){
      column = X[, col]
      X[, col] = (column*se.x[col]) + mean.x[col]
    }
    y = (y * se.y) + mean.y
    b = (beta * se.y)/ se.x
  }else{
    b = beta
  }
  b0 =  sum(y-X%*%b)/n
  return(c(b0,b))
}

# Split data set into training and testing

train.x = as.matrix(df1[!(row.names(df1) %in% test.PID), ])
test.x =  as.matrix(df1[row.names(df1) %in% test.PID, ])



beta.lasso = mylasso(train.x, train.y, lam = 80, standardize = T)
Ytest.pred = (test.x %*% matrix(beta.lasso[-1], nrow = length(beta.lasso[-1])))+matrix(rep(beta.lasso[1], nrow(test.x)), nrow = nrow(test.x))
sub3 = data.frame(cbind(test.PID, round(Ytest.pred, 1)))
colnames(sub3) = c("PID", "Sale_Price")
write.table(sub3, file="mysubmission3.txt", quote=F, row.names=F, sep=",")
