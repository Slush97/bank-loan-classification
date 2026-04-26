# bank_loan.R - Universal Bank personal loan classification
# Models: logistic regression, decision tree, ANN
# Methodology follows Lab 6 (LR + stepwise), Lab 7 (CART), Lab 8 (ANN)

# install.packages("caret")           # uncomment if not installed
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("nnet")
# install.packages("NeuralNetTools")
# install.packages("ROSE")
# install.packages("ggplot2")
# install.packages("gridExtra")

library(caret)
library(rpart)
library(rpart.plot)
library(nnet)
library(NeuralNetTools)
library(ROSE)
library(ggplot2)
library(gridExtra)

set.seed(123)


# load
bank <- read.csv("UniversalBank.csv", header = TRUE, check.names = FALSE)
str(bank)
head(bank)
summary(bank)

# distributions of the four most variable continuous features
h1 <- ggplot(bank, aes(Income))   + geom_histogram(bins = 30) + labs(title = "Income",   x = "$K")       + theme_minimal()
h2 <- ggplot(bank, aes(CCAvg))    + geom_histogram(bins = 30) + labs(title = "CCAvg",    x = "$K/month") + theme_minimal()
h3 <- ggplot(bank, aes(Age))      + geom_histogram(bins = 30) + labs(title = "Age",      x = "years")    + theme_minimal()
h4 <- ggplot(bank, aes(Mortgage)) + geom_histogram(bins = 30) + labs(title = "Mortgage", x = "$K")       + theme_minimal()
grid.arrange(h1, h2, h3, h4, ncol = 2)


# cleaning
bank$ID         <- NULL                    # row counter
bank$`ZIP Code` <- NULL                    # too many distinct values
names(bank) <- gsub(" ", "_", names(bank))
bank$Experience[bank$Experience < 0] <- 0  # 52 rows are negative, clip to 0
sum(is.na(bank))                           # 0 - no NAs

bank$Personal_Loan <- factor(bank$Personal_Loan, levels = c(0, 1))
table(bank$Personal_Loan)
prop.table(table(bank$Personal_Loan))      # ~9.6% acceptance, very imbalanced

# treat the obvious categoricals as factors so LR doesn't assume linear spacing
bank$Education          <- factor(bank$Education, levels = c(1, 2, 3),
                                  labels = c("Undergrad","Graduate","Advanced"))
bank$Family             <- factor(bank$Family, levels = c(1, 2, 3, 4))
bank$Securities_Account <- factor(bank$Securities_Account, levels = c(0, 1))
bank$CD_Account         <- factor(bank$CD_Account, levels = c(0, 1))
bank$Online             <- factor(bank$Online, levels = c(0, 1))
bank$CreditCard         <- factor(bank$CreditCard, levels = c(0, 1))

# class distribution
ggplot(bank, aes(Personal_Loan, fill = Personal_Loan)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Personal Loan class distribution",
       x = "Personal Loan", y = "Number of customers") +
  theme_minimal() + theme(legend.position = "none")

# Income vs CCAvg colored by acceptance
ggplot(bank, aes(Income, CCAvg, color = Personal_Loan)) +
  geom_point(alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Income vs CCAvg by Loan Acceptance",
       x = "Income ($K)", y = "CCAvg ($K/month)",
       color = "Personal Loan") +
  theme_minimal()

# Income by Education, split by acceptance
ggplot(bank, aes(Education, Income, fill = Personal_Loan)) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Income by Education and Loan Acceptance",
       x = "Education", y = "Income ($K)", fill = "Personal Loan") +
  theme_minimal()


# train / test split (70/30, stratified on Personal_Loan)
# stratified split preserves the 9.6% acceptance rate in both folds so the
# test acceptor count does not swing run-to-run with this much imbalance.
idx   <- createDataPartition(bank$Personal_Loan, p = 0.7, list = FALSE)
train <- bank[idx,  ]
test  <- bank[-idx, ]
table(train$Personal_Loan)
table(test$Personal_Loan)


# forward stepwise feature selection (Lab 6)
null_mod <- glm(Personal_Loan ~ 1, data = train, family = binomial)
full_mod <- glm(Personal_Loan ~ ., data = train, family = binomial)

step_forw <- step(null_mod,
                  scope = list(lower = null_mod, upper = full_mod),
                  direction = "forward",
                  trace = 0)
step_forw
summary(step_forw)

selected <- attr(terms(step_forw), "term.labels")
selected
sel_formula <- as.formula(paste("Personal_Loan ~", paste(selected, collapse = " + ")))


# logistic regression (Lab 6) - step_forw is already a glm, use it directly
lr_mod <- step_forw

prob_lr <- predict(lr_mod, newdata = test, type = "response")
pred_lr <- factor(ifelse(prob_lr >= 0.5, "1", "0"),
                  levels = levels(test$Personal_Loan))
cm_lr <- confusionMatrix(pred_lr, test$Personal_Loan, positive = "1")
cm_lr


# decision tree (Lab 7) - grow deep then prune at min xerror cp
dt_mod <- rpart(sel_formula, data = train, method = "class",
                control = rpart.control(minsplit = 20, cp = 0.001))
best_cp   <- dt_mod$cptable[which.min(dt_mod$cptable[,"xerror"]), "CP"]
dt_pruned <- prune(dt_mod, cp = best_cp)

rpart.plot(dt_pruned, type = 2, extra = 104, fallen.leaves = TRUE,
           cex = 0.7, box.palette = "Blues",
           main = "Decision Tree (pruned)")

pred_dt <- predict(dt_pruned, newdata = test, type = "class")
pred_dt <- factor(pred_dt, levels = levels(test$Personal_Loan))
cm_dt   <- confusionMatrix(pred_dt, test$Personal_Loan, positive = "1")
cm_dt


# ANN (Lab 8)
# 1) oversample TRAIN only (anti-leakage)
# 2) scale predictors with TRAIN means/SDs, reapply same numbers to TEST
# 3) fit nnet, predict, threshold at 0.5

train_sel <- train[, c(selected, "Personal_Loan")]
test_sel  <- test[ , c(selected, "Personal_Loan")]

train_ann <- ovun.sample(Personal_Loan ~ ., data = train_sel,
                         method = "over", seed = 123)$data
table(train_ann$Personal_Loan)

# factors -> 0/1 dummy columns so nnet has a numeric matrix to chew on
x_train_raw <- model.matrix(Personal_Loan ~ . - 1, data = train_ann)
x_test_raw  <- model.matrix(Personal_Loan ~ . - 1, data = test_sel)

mu       <- colMeans(x_train_raw)
sd_train <- apply(x_train_raw, 2, sd)
x_train  <- scale(x_train_raw, center = mu, scale = sd_train)
x_test   <- scale(x_test_raw,  center = mu, scale = sd_train)

y_train <- as.numeric(train_ann$Personal_Loan) - 1   # factor "0"/"1" -> 0/1

ann_mod <- nnet(x = x_train, y = y_train,
                size = 5, decay = 0.001, maxit = 1000,
                linout = FALSE, trace = FALSE)
ann_mod$n
length(ann_mod$wts)
summary(ann_mod)

# visualise the trained network
par(mar = c(2, 8, 2, 2))
plotnet(ann_mod, alpha = 0.7,
        circle_col = "lightyellow",
        pos_col = "steelblue", neg_col = "tomato",
        bias = TRUE, node_labs = TRUE, var_labs = TRUE,
        pad_x = 0.7)

prob_ann <- predict(ann_mod, newdata = x_test)
pred_ann <- factor(ifelse(prob_ann >= 0.5, "1", "0"),
                   levels = levels(test$Personal_Loan))
cm_ann <- confusionMatrix(pred_ann, test$Personal_Loan, positive = "1")
cm_ann


# compare the three models
results <- rbind(
  LR  = c(cm_lr$overall["Accuracy"],  cm_lr$byClass[c("Precision","Recall","F1")]),
  DT  = c(cm_dt$overall["Accuracy"],  cm_dt$byClass[c("Precision","Recall","F1")]),
  ANN = c(cm_ann$overall["Accuracy"], cm_ann$byClass[c("Precision","Recall","F1")])
)
print(round(results, 4))
