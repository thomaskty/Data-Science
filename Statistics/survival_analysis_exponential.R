library(survival)

# Hard-coded sample data (20 rows)
health_dat <- data.frame(
  id = 1:20,
  time_months = c(4, 6, 8, 5, 10, 12, 7, 9, 3, 11, 14, 13, 6, 15, 16, 5, 8, 10, 12, 7),
  event = c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1),
  age = c(52, 60, 48, 66, 55, 63, 50, 58, 45, 61, 67, 59, 54, 64, 69, 53, 57, 62, 56, 51),
  treatment = factor(
    c("Standard", "Standard", "NewDrug", "Standard", "NewDrug", "NewDrug", "Standard", "NewDrug", "Standard", "NewDrug", "Standard", "NewDrug", "Standard", "NewDrug", "Standard", "Standard", "NewDrug", "NewDrug", "Standard", "NewDrug"),
    levels = c("Standard", "NewDrug")
  )
)

# Exponential AFT model
exp_fit <- survreg(
  Surv(time_months, event) ~ age + treatment,
  data = health_dat,
  dist = "exponential"
)

cat("--- Exponential Survival Model ---\n")
print(summary(exp_fit))

# Coefficient table with hazard-ratio view
coef_table <- as.data.frame(summary(exp_fit)$table)
coef_table$variable <- rownames(coef_table)
rownames(coef_table) <- NULL

coef_table$hazard_ratio <- exp(-coef_table$Value)
coef_table$hr_ci_lower_95 <- exp(-(coef_table$Value + 1.96 * coef_table$`Std. Error`))
coef_table$hr_ci_upper_95 <- exp(-(coef_table$Value - 1.96 * coef_table$`Std. Error`))

coef_out <- coef_table[, c("variable", "Value", "Std. Error", "z", "p", "hazard_ratio", "hr_ci_lower_95", "hr_ci_upper_95")]
coef_out$Value <- round(coef_out$Value, 4)
coef_out$`Std. Error` <- round(coef_out$`Std. Error`, 4)
coef_out$z <- round(coef_out$z, 3)
coef_out$p <- round(coef_out$p, 4)
coef_out$hazard_ratio <- round(coef_out$hazard_ratio, 3)
coef_out$hr_ci_lower_95 <- round(coef_out$hr_ci_lower_95, 3)
coef_out$hr_ci_upper_95 <- round(coef_out$hr_ci_upper_95, 3)

cat("\n--- Coefficient Table (coef_out) ---\n")
print(coef_out)

# Compact treatment group hazard ratio
beta_trt <- coef(exp_fit)["treatmentNewDrug"]
se_trt <- sqrt(vcov(exp_fit)["treatmentNewDrug", "treatmentNewDrug"])
hr_trt <- exp(-beta_trt)
hr_trt_ci <- c(
  exp(-(beta_trt + 1.96 * se_trt)),
  exp(-(beta_trt - 1.96 * se_trt))
)

cat("\n--- Treatment Group Hazard Ratio ---\n")
cat(
  "NewDrug vs Standard HR =", round(hr_trt, 3),
  "(95% CI:", round(hr_trt_ci[1], 3), "to", round(hr_trt_ci[2], 3), ")\n"
)


# Example predictions for new patients
new_profiles <- data.frame(
  age = c(50, 60, 65),
  treatment = factor(c("Standard", "NewDrug", "NewDrug"), levels = levels(health_dat$treatment))
)

lp <- predict(exp_fit, newdata = new_profiles, type = "lp")
lambda <- exp(-lp)  # constant hazard for each profile

pred_out <- cbind(
  new_profiles,
  hazard_per_month = round(lambda, 4),
  survival_prob_6m = round(exp(-lambda * 6), 4),
  event_prob_12m = round(1 - exp(-lambda * 12), 4),
  median_survival_months = round(log(2) / lambda, 2)
)

cat("\n--- Example Predictions ---\n")
print(pred_out)
