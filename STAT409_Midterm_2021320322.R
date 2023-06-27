# 1 - (f)

crd <- read.csv('crd.csv')
attach(crd)

y <- c(Group1, Group2, Group3, Group4, Group5)
X <- matrix(rep(0, 250), nrow = 50, ncol = 5)
for (i in 1:5) {
  X[(i*10 - 9 + (i-1)*50) : (i*10 + (i-1)*50)] <- rep(1, 10)
}

obj <- qr(X)
Q <- qr.Q(obj, complete = T)
R <- matrix(0, 5, 5)
R[upper.tri(R, diag = T)] <- obj$qr[upper.tri(obj$qr, diag = T)]

z <- crossprod(Q, y)
z1 <- z[1:5]
z2 <- z[-c(1:5)]

mu.hat <- backsolve(R, z1)
y.hat <- Q %*% c(z1, rep(0, length(z2)))
resid <- Q %*% c(rep(0, length(z1)), z2)

SStrt <- sum((y.hat - mean(mu.hat)) ^ 2)
SSE <- sum(resid ^ 2)

MStrt <- SStrt / (5 - 1)
MSE <- SSE / (5 * (10 - 1))
F_stat <- MStrt / MSE
c(MStrt, MSE, F_stat)



# 2 - (a)
library(tidyverse)

train <- read.csv('train.csv')
test <- read.csv('test.csv')

polyr1 <- lm(y ~ poly(x, 1), data = train)
polyr2 <- lm(y ~ poly(x, 2), data = train)
polyr3 <- lm(y ~ poly(x, 3), data = train)
polyr4 <- lm(y ~ poly(x, 4), data = train)
polyr5 <- lm(y ~ poly(x, 5), data = train)
polyr6 <- lm(y ~ poly(x, 6), data = train)
polyr7 <- lm(y ~ poly(x, 7), data = train)
polyr8 <- lm(y ~ poly(x, 8), data = train)
polyr9 <- lm(y ~ poly(x, 9), data = train)
polyr10 <- lm(y ~ poly(x, 10), data = train)

data.plot <- tibble(x = rep(test$x, 10))
data.plot$pred <- c(predict(polyr1, test),
                    predict(polyr2, test),
                    predict(polyr3, test),
                    predict(polyr4, test),
                    predict(polyr5, test),
                    predict(polyr6, test),
                    predict(polyr7, test),
                    predict(polyr8, test),
                    predict(polyr9, test),
                    predict(polyr10, test))
data.plot$index <- c(rep("p=1", 500),
                     rep("p=2", 500),
                     rep("p=3", 500),
                     rep("p=4", 500),
                     rep("p=5", 500),
                     rep("p=6", 500),
                     rep("p=7", 500),
                     rep("p=8", 500),
                     rep("p=9", 500),
                     rep("p=10", 500))

library(gridExtra)

train_plot <- ggplot(train, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index)) +
  ggtitle('Training set')
test_plot <-  ggplot(test, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index)) +
  ggtitle('Testing set')

grid.arrange(train_plot, test_plot, ncol = 2)



# 2 - (b)
R2 <- c(summary(polyr1)$r.squared,
        summary(polyr2)$r.squared,
        summary(polyr3)$r.squared,
        summary(polyr4)$r.squared,
        summary(polyr5)$r.squared,
        summary(polyr6)$r.squared,
        summary(polyr7)$r.squared,
        summary(polyr8)$r.squared,
        summary(polyr9)$r.squared,
        summary(polyr10)$r.squared)
ggplot(mapping = aes(x = 1:10, y = R2)) +
  geom_point() +
  geom_line() +
  xlab('p') +
  scale_x_continuous(breaks = 1:10)



# 2 - (d)
bic <- function(x, p) {
  return (sum(summary(x)$residuals ^ 2) + log(500) * p)
}
BICs <- c(bic(polyr1, 1), bic(polyr2, 2), bic(polyr3, 3), bic(polyr4, 4), bic(polyr5, 5), 
          bic(polyr6, 6), bic(polyr7, 7), bic(polyr8, 8), bic(polyr9, 9), bic(polyr10, 10))
which.min(BICs) # BIC에 근거했을 때 p = 5인 모델을 선택



# 2 - (e)
library(glmnet)

lasso <- glmnet(poly(train$x, 10), train$y, alpha = 1)
ridge <- glmnet(poly(train$x, 10), train$y, alpha = 0)

penalized_est <- function(obj, x, y, alpha) {
  grid <- obj$lambda
  cv <- cv.glmnet(x, y, alpha = alpha, lambda = grid)
  opt.lambda <- cv$lambda.min
  opt.id <- which(near(grid,opt.lambda))
  coef <- coef(obj)[,opt.id]
  list <- list(lambda = opt.lambda, coef = coef)
  return (list)
}

opt.lasso <- penalized_est(lasso, poly(train$x, 10), train$y, 1)
opt.ridge <- penalized_est(ridge, poly(train$x, 10), train$y, 0)

obj.lasso <- glmnet(poly(train$x, 10), train$y, alpha = 1, lambda = opt.lasso$lambda)
obj.ridge <- glmnet(poly(train$x, 10), train$y, alpha = 0, lambda = opt.ridge$lambda)

data.plot <- tibble(x = rep(test$x, 2))
data.plot$pred <- c(predict(obj.lasso, newx = poly(test$x, 10)),
                    predict(obj.ridge, newx = poly(test$x, 10)))
data.plot$index <- c(rep("LASSO", 500),
                     rep("Ridge", 500))

train_plot <- ggplot(train, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index),
            size = 1.5) +
  ggtitle('Training set')
test_plot <- ggplot(test, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index),
            size = 1.5) +
  ggtitle('Testing set')

grid.arrange(train_plot, test_plot, ncol = 2)



# 2 - (f)
library(splines)

BICs <- c()
for (i in 3:30) {
  spliner <- lm(y ~ bs(x, df = i), data = train)
  BICs <- c(BICs, bic(spliner, i))
}
which.min(BICs) # BIC에 근거했을 때 df = 5인 모델을 선택

spliner <- lm(y ~ bs(x, df = 5), data = train)

data.plot <- tibble(x = test$x)
data.plot$pred <- c(predict(spliner, test))
data.plot$index <- c(rep("df=5", 500))

train_plot <- ggplot(train, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index),
            size = 1.5) +
  ggtitle('Training set')
test_plot <- ggplot(test, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index),
            size = 1.5) +
  ggtitle('Testing set')

grid.arrange(train_plot, test_plot, ncol = 2)



# 2 - (g)
library(quantreg)

BICs <- c()
for (i in 3:30) {
  spliner <- rq(y ~ bs(x, df = i), tau = 0.5, data = train)
  BICs <- c(BICs, bic(spliner, i))
}
which.min(BICs) # BIC에 근거했을 때 df = 5인 모델을 선택

spliner <- rq(y ~ bs(x, df = 5), tau = 0.5, data = train)

data.plot <- tibble(x = test$x)
data.plot$pred <- c(predict(spliner, test))
data.plot$index <- c(rep("df=5", 500))

train_plot <- ggplot(train, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index),
            size = 1.5) +
  ggtitle('Training set')
test_plot <- ggplot(test, aes(x, y)) +
  geom_point(alpha = 0.3) +
  geom_line(data = data.plot,
            mapping = aes(x, pred,
                          col = index, linetype = index),
            size = 1.5) +
  ggtitle('Testing set')

grid.arrange(train_plot, test_plot, ncol = 2)



# 2 - (h)

test_poly <- mean((test$y - predict(polyr5, test)) ^ 2)
test_lasso <- mean((test$y - predict(obj.lasso, newx = poly(test$x, 10))) ^ 2)
test_ridge <- mean((test$y - predict(obj.ridge, newx = poly(test$x, 10))) ^ 2)
test_bsmean <- mean((test$y - predict(lm(y ~ bs(x, df = 5), data = train), test)) ^ 2)
test_bsmed <- mean((test$y - predict(rq(y ~ bs(x, df = 5), tau = 0.5, data = train), test)) ^ 2)
c(test_poly, test_lasso, test_ridge, test_bsmean, test_bsmed)

