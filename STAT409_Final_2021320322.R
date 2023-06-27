setwd("C:/Users/BDK/OneDrive/Desktop/통계적데이터과학II")

# 3.
train <- read.csv("train.csv")
test <- read.csv("test.csv")

accuracy <- c()

# Logistic Regression
set.seed(123)
train.1 <- train
train.1$y[which(train$y == -1)] <- 0
test.1 <- test
test.1$y[which(test$y == -1)] <- 0
obj.1 <- glm(y ~ ., data = train.1, family = 'binomial')

pred.1 <- predict(obj.1, test.1)
tab.1 <- table(test.1$y, pred.1 > 0.5)
accuracy <- c(accuracy, (tab.1[1, 1] + tab.1[2, 2]) / length(test$y))
accuracy

# LASSO-penalized Logistic Regression
set.seed(123)
library(glmnet)
library(dplyr)

train.x <- model.matrix(y ~ ., data = train.1)[,-1]
test.x <- model.matrix(y ~ ., data = test.1)[,-1]
lasso <- glmnet(train.x, train.1$y, alpha = 1, family = 'binomial')

penalized_est <- function(obj, x, y, alpha) {
  grid <- obj$lambda
  cv <- cv.glmnet(x, y, alpha = alpha, lambda = grid, nfolds = 5)
  opt.lambda <- cv$lambda.min
  opt.id <- which(near(grid,opt.lambda))
  coef <- coef(obj)[,opt.id]
  list <- list(lambda = opt.lambda, coef = coef)
  return (list)
}

opt.lasso <- penalized_est(lasso, train.x, train.1$y, 1)
obj.2 <- glmnet(train.x, train.1$y, alpha = 1, lambda = opt.lasso$lambda)
pred.2 <- predict(obj.2, newx = test.x)
tab.2 <- table(test.1$y, pred.2 > 0.5)
accuracy <- c(accuracy, (tab.2[1, 1] + tab.2[2, 2]) / length(test$y))
accuracy

# Linear SVM
set.seed(123)
library(kernlab)
library(caret)

train.2 <- train
train.2$y <- as.factor(train.2$y)
test.2 <- test
test.2$y <- as.factor(test.2$y)

train_control <- trainControl(method = 'repeatedcv', number = 5, repeats = 5)
svm <- train(y ~ ., data = train.2, method = 'svmLinear', trControl = train_control,
             metric = 'Accuracy', tuneGrid = expand.grid(C = seq(0.1, 3, length = 30)))
svm$bestTune

obj.3 <- ksvm(y ~ ., data = train, type = 'C-svc',
              kernel = 'vanilladot', cross = 5, C = 0.2)
pred.3 <- predict(obj.3, test.x)
tab.3 <- table(test$y, pred.3)
accuracy <- c(accuracy, (tab.3[1, 1] + tab.3[2, 2]) / length(test$y))
accuracy

# Gaussian Kernel SVM
set.seed(123)

train_control <- trainControl(method = 'repeatedcv', number = 5, repeats = 5)
param_grid = expand.grid(C = seq(0.1, 3, 0.1), sigma = seq(0.01, 0.05, 0.01))
svm_rbf <- train(y ~ ., data = train.2, method = 'svmRadial', trControl = train_control,
                 metric = 'Accuracy', tuneGrid = param_grid)
svm_rbf$bestTune

obj.4 <- ksvm(y ~ ., data = train, type = 'C-svc',
              kernel = 'rbfdot', cross = 5, sigma = 0.01, C = 1.7)
pred.4 <- predict(obj.4, test.x)
tab.4 <- table(test$y, pred.4)
accuracy <- c(accuracy, (tab.4[1, 1] + tab.4[2, 2]) / length(test$y))
accuracy

# Classification Tree
set.seed(123)

library(tree)
obj.5 <- tree(y ~ ., data = train.2)
cv.obj <- cv.tree(obj.5, FUN = prune.misclass)
size <- cv.obj$size
error <- cv.obj$dev
lambda <- cv.obj$k
best.size <- size[which.min(error)]

plot(size, error, type = "b",
     xlab = "# of terminal nodes", ylab = "CV errors")
abline(v = best.size, col = 2)
prune.obj <- prune.misclass(obj.5, best = best.size)
plot(prune.obj); text(prune.obj, pretty = 0)

pred.5 <- predict(prune.obj, test.2, type = 'class')
tab.5 <- table(test$y, pred.5)
accuracy <- c(accuracy, (tab.5[1, 1] + tab.5[2, 2]) / length(test$y))
accuracy

# Random Forest
set.seed(123)
library(randomForest)

train_control <- trainControl(method = 'repeatedcv', number = 5, repeats = 5)
best_mtry <- c()
accuracy.rf <- c()
for (ntree in c(100, 300, 500)) {
  rf <- train(y ~ ., data = train.2, method = 'rf', trControl = train_control,
            metric = 'Accuracy', tuneGrid = expand.grid(.mtry = 1:20), ntree = ntree)
  best_mtry <- c(best_mtry, rf$bestTune[1, 1])
  accuracy.rf <- c(accuracy, max(rf$results[, 2]))
}

which.max(accuracy.rf)
best_mtry[which.max(accuracy.rf)]
  
obj.6 <- randomForest(y ~ ., data = train.2, mtry = 11, ntree = 300)
pred.6 <- predict(obj.6, test.2, type = 'class')
tab.6 <- table(test$y, pred.6)
accuracy <- c(accuracy, (tab.6[1, 1] + tab.6[2, 2]) / length(test$y))
accuracy

# Logit Boosting
set.seed(123)
library(gbm)

obj.7 <- gbm(y ~ ., data = train.1, distribution = "bernoulli",
             n.trees = 500, cv.folds = 5)
opt.size <- gbm.perf(obj.7, method = 'cv')
pred.7 <- predict(obj.7, test.1, n.trees = opt.size)
tab.7 <- table(test$y, (sign(pred.7) + 1) / 2)
accuracy <- c(accuracy, (tab.7[1, 1] + tab.7[2, 2]) / length(test$y))
accuracy

accuracy <- matrix(accuracy, nrow = 7)
rownames(accuracy) <- c('Logistic Regression',
                        'LASSO-penalized Logistic Regression',
                        'Linear SVM',
                        'Gaussian Kernel SVM',
                        'Classification Tree',
                        'Random Forest',
                        'Logit Boosting')
colnames(accuracy) <- c('Accuracy')
accuracy


# 4.
set.seed(123)
library(tidyverse)
library(ggplot2)

train <- read.csv("data_usv.csv")
pca <- prcomp(train)
pc1 <- pca$x[,1]
pc2 <- pca$x[,2]
pc <- data.frame(x = pc1, y = pc2)
ggplot(aes(x = x, y = y), data = pc) +
  geom_point()

library(Rtsne)
t_sne <- Rtsne(train)
tsne <- data.frame(x = t_sne$Y[,1], y = t_sne$Y[,2])
ggplot(aes(x = x, y = y), data = tsne) +
  geom_point()

pca_kmeans <- kmeans(pc, centers = 3, iter.max = 10000)
plot1 <- ggplot(aes(x = x, y = y), data = pc) +
  geom_point(color = pca_kmeans$cluster) +
  ggtitle('PCA + k-Means')

tsne_kmeans <- kmeans(tsne, centers = 3, iter.max = 10000)
plot2 <- ggplot(aes(x = x, y = y), data = tsne) +
  geom_point(color = tsne_kmeans$cluster) +
  ggtitle('t-SNE + k-Means')

pca_hc <- hclust(d = dist(pc), method = 'complete')
dend1 <- plot(pca_hc, hang = -1)
rect.hclust(pca_hc, k = 3)
plot3 <- ggplot(aes(x = x, y = y), data = pc) +
  geom_point(color = cutree(pca_hc, k = 3)) +
  ggtitle('PCA + HClust')
plot3

tsne_hc <- hclust(d = dist(tsne), method = 'complete')
dend2 <- plot(tsne_hc, hang = -1)
rect.hclust(tsne_hc, k = 3)
plot4 <- ggplot(aes(x = x, y = y), data = tsne) +
  geom_point(color = cutree(tsne_hc, k = 3)) +
  ggtitle('t-SNE + HClust')

library(mclust)
pca_gmm <- Mclust(pc, G = 3)
plot5 <- ggplot(aes(x = x, y = y), data = pc) +
  geom_point(color = pca_gmm$classification) +
  ggtitle('PCA + GMix')

tsne_gmm <- Mclust(tsne, G = 3)
plot6 <- ggplot(aes(x = x, y = y), data = tsne) +
  geom_point(color = tsne_gmm$classification) +
  ggtitle('t-SNE + GMix')

library(dbscan)
pca_db <- dbscan(pc, eps = 1, minPts = 4)
pca_db$cluster[which(pca_db$cluster == 0)] <- max(pca_db$cluster) + 1
plot7 <- ggplot(aes(x = x, y = y), data = pc) +
  geom_point(color = pca_db$cluster) +
  ggtitle('PCA + dbSCAN')

tsne_db <- dbscan(tsne, eps = 2.5, minPts = 5)
tsne_db$cluster[which(tsne_db$cluster == 0)] <- max(tsne_db$cluster) + 1
plot8 <- ggplot(aes(x = x, y = y), data = tsne) +
  geom_point(color = tsne_db$cluster) +
  ggtitle('t-SNE + dbSCAN')

library(gridExtra)
grid.arrange(plot1, plot2, plot3, plot4,
             plot5, plot6, plot7, plot8, ncol = 2)