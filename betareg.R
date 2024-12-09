library(betareg)
library(dplyr)

df.train <- read.csv("./Data/test.csv")
df.test <- read.csv("./Data/train.csv")

set.seed(42)
beta_res <- betareg(Cover ~ Site_id + Visit_id + Species_id, data = df.train, link = "logit")
print(summary(beta_res))
coefs <- coefficients(beta_res)
coefs <- c(coefs[-c(1, length(coefs))], coefs[1], coefs[length(coefs)])
write(coefs, file = "betareg_res.dat")

beta_pred <- predict(beta_res, df.test[, -4])
cat("RMSE: ", sqrt(mean((beta_pred - df.test$Cover)^2)), "\n")

# lin <- lm(Cover ~ Site_id + Visit_id + Species_id, data = df.train)
# summary(lin)
# BIC(lin)
