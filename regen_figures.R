# Regenerate r_figures/*.png from the current bank_loan.R run.
# Mirrors the original chart styling from bank_loan.R / make_report.R --
# default ggplot theme_minimal with RColorBrewer palettes.

suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
  library(rpart.plot)
  library(NeuralNetTools)
  library(caret)
})

tmp_pdf <- tempfile(fileext = ".pdf")
pdf(tmp_pdf)
invisible(capture.output(source("bank_loan.R")))
invisible(dev.off())
unlink(tmp_pdf)

figdir <- "../OM 484 Materials/r_figures"
dir.create(figdir, showWarnings = FALSE, recursive = TRUE)

# relabel target factor for plotting legends
bank$Personal_Loan <- factor(bank$Personal_Loan,
                             levels = c("0", "1"),
                             labels = c("Declined", "Accepted"))

# class distribution
ggplot(bank, aes(Personal_Loan, fill = Personal_Loan)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Personal Loan class distribution",
       x = "Personal Loan", y = "Number of customers") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave(file.path(figdir, "class_dist.png"), width = 6, height = 4, dpi = 200)

# four numeric histograms
h1 <- ggplot(bank, aes(Income))   + geom_histogram(bins = 30) + labs(title = "Income",   x = "$K")       + theme_minimal()
h2 <- ggplot(bank, aes(CCAvg))    + geom_histogram(bins = 30) + labs(title = "CCAvg",    x = "$K/month") + theme_minimal()
h3 <- ggplot(bank, aes(Age))      + geom_histogram(bins = 30) + labs(title = "Age",      x = "years")    + theme_minimal()
h4 <- ggplot(bank, aes(Mortgage)) + geom_histogram(bins = 30) + labs(title = "Mortgage", x = "$K")       + theme_minimal()
ggsave(file.path(figdir, "histograms.png"),
       arrangeGrob(h1, h2, h3, h4, ncol = 2),
       width = 7, height = 5, dpi = 200)

# Income vs CCAvg colored by acceptance
ggplot(bank, aes(Income, CCAvg, color = Personal_Loan)) +
  geom_point(alpha = 0.5) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Income vs CCAvg by Loan Acceptance",
       x = "Income ($K)", y = "CCAvg ($K/month)",
       color = "Personal Loan") +
  theme_minimal()
ggsave(file.path(figdir, "scatter_income_ccavg.png"), width = 6, height = 4, dpi = 200)

# Income by Education, split by acceptance
ggplot(bank, aes(Education, Income, fill = Personal_Loan)) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Income by Education and Loan Acceptance",
       x = "Education", y = "Income ($K)", fill = "Personal Loan") +
  theme_minimal()
ggsave(file.path(figdir, "box_income_edu.png"), width = 6, height = 4, dpi = 200)

# Pruned decision tree
png(file.path(figdir, "tree.png"), width = 8, height = 5, units = "in", res = 200)
rpart.plot(dt_pruned, type = 2, extra = 104, fallen.leaves = TRUE,
           cex = 0.7, box.palette = "Blues",
           main = "Decision Tree (pruned)")
dev.off()

# ANN architecture
png(file.path(figdir, "ann_architecture.png"), width = 10, height = 6, units = "in", res = 200)
par(mar = c(2, 8, 2, 2))
plotnet(ann_mod, alpha = 0.7,
        circle_col = "lightyellow",
        pos_col = "steelblue", neg_col = "tomato",
        bias = TRUE, node_labs = TRUE, var_labs = TRUE,
        pad_x = 0.7)
dev.off()

# confusion matrices for the three models
cm_to_df <- function(cm, model_name) {
  m <- cm$table
  data.frame(
    Predicted = factor(rep(c("Declined","Accepted"), each = 2),
                       levels = c("Accepted","Declined")),
    Actual    = factor(rep(c("Declined","Accepted"), times = 2),
                       levels = c("Declined","Accepted")),
    Count     = c(m["0","0"], m["0","1"], m["1","0"], m["1","1"]),
    Model     = model_name,
    Correct   = c(TRUE, FALSE, FALSE, TRUE)
  )
}

cm_df <- rbind(
  cm_to_df(cm_lr,  "Logistic Regression"),
  cm_to_df(cm_dt,  "Decision Tree"),
  cm_to_df(cm_ann, "Neural Network")
)
cm_df$Model <- factor(cm_df$Model,
                      levels = c("Logistic Regression","Decision Tree","Neural Network"))

ggplot(cm_df, aes(Actual, Predicted, fill = Correct)) +
  geom_tile(color = "grey60", linewidth = 0.4) +
  geom_text(aes(label = Count), size = 5, fontface = "bold") +
  scale_fill_manual(values = c(`TRUE` = "#8AB4F8", `FALSE` = "#FCDDDB"),
                    guide = "none") +
  facet_wrap(~ Model, nrow = 1) +
  labs(x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(),
        strip.text = element_text(face = "bold", size = 12),
        aspect.ratio = 1)
ggsave(file.path(figdir, "confusion_matrices.png"),
       width = 9, height = 3.5, dpi = 200)

cat("PNGs written to:", normalizePath(figdir), "\n")
