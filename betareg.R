library(betareg)
library(dplyr)

df <- read.csv("./Data/observations.csv", sep=";") |> select(-Cover_class)

df["Cover"] <- df["Cover"] / 100
df[,-4] <- scale(df[,-4])

set.seed(5)
train_idx <- sample(1:nrow(df), size = as.integer(0.33 * nrow(df)))

df.train <- df[train_idx,]
df.test <- df[-train_idx,]
beta_res <- betareg(Cover ~ Site_id + Visit_id + Species_id, data = df.train, link="logit")
summary(beta_res)

beta_pred <- predict(beta_res, df.test[,-4])
sqrt(mean((beta_pred - df.test$Cover)^2))
