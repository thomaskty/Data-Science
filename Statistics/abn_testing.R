############################################
# 🧪 A/B/n TESTING LAB (5 GROUPS)
############################################

# Libraries
library(dplyr)
library(ggplot2)

############################################
# 📊 STEP 1: CREATE REALISTIC DATASET
############################################

# 5 groups: A (control), B, C, D, E (variants)
group <- c(
  rep("A", 60),
  rep("B", 60),
  rep("C", 60),
  rep("D", 60),
  rep("E", 60)
)

# Conversion (realistic pattern: gradual improvement)
conversion <- c(
  
  # A (baseline ~15%)
  0,0,1,0,0,1,0,0,1,0, 0,1,0,0,0,1,0,0,1,0,
  0,0,1,0,0,1,0,0,0,1, 0,0,1,0,0,1,0,0,0,1,
  0,0,1,0,0,1,0,0,1,0, 0,0,1,0,0,0,1,0,0,0,
  
  # B (~20%)
  0,1,1,0,0,1,0,1,1,0, 0,1,1,0,1,1,0,1,1,0,
  1,0,1,0,1,1,0,1,0,1, 0,1,1,0,1,1,0,1,1,0,
  1,0,1,1,0,1,1,0,1,1, 0,1,1,0,1,0,1,0,1,0,
  
  # C (~25%)
  1,0,1,1,0,1,1,0,1,1, 0,1,1,0,1,1,0,1,1,0,
  1,1,0,1,1,0,1,1,0,1, 1,0,1,1,0,1,1,0,1,1,
  0,1,1,0,1,1,0,1,1,0, 1,1,0,1,1,0,1,1,0,1,
  
  # D (~30%)
  1,1,1,0,1,1,1,0,1,1, 0,1,1,1,0,1,1,1,0,1,
  1,1,0,1,1,1,0,1,1,1, 0,1,1,1,0,1,1,1,0,1,
  1,1,1,0,1,1,1,0,1,1, 0,1,1,1,0,1,1,0,1,1,
  
  # E (~35% best performer)
  1,1,1,1,0,1,1,1,1,0, 1,1,1,1,0,1,1,1,1,0,
  1,1,1,0,1,1,1,0,1,1, 1,1,0,1,1,1,0,1,1,1,
  1,1,1,1,0,1,1,1,1,0, 1,1,1,0,1,1,1,0,1,1
)

# Revenue (only if conversion = 1, varies by group)
set.seed(123)

revenue <- ifelse(conversion == 1,
                  round(rnorm(length(conversion), mean = 100, sd = 20), 2),
                  0)

# Create dataframe
df <- data.frame(group, conversion, revenue)

############################################
# 📊 STEP 2: EXPLORATORY ANALYSIS
############################################

summary_table <- df %>%
  group_by(group) %>%
  summarise(
    users = n(),
    conversions = sum(conversion),
    conversion_rate = mean(conversion),
    avg_revenue = mean(revenue)
  )

print(summary_table)

############################################
# 📈 STEP 3: VISUALIZATION
############################################

# Conversion Rate Plot
ggplot(summary_table, aes(x = group, y = conversion_rate, fill = group)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ylim(0,1) +
  labs(title = "Conversion Rate by Group")

# Revenue Plot
ggplot(df, aes(x = group, y = revenue, fill = group)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Revenue Distribution by Group")

############################################
# 🧠 STEP 4: A/B/n TEST (CONVERSION)
############################################

conv_summary <- df %>%
  group_by(group) %>%
  summarise(conversions = sum(conversion),
            total = n())

x <- conv_summary$conversions
n <- conv_summary$total

prop_test <- prop.test(x, n)

print(prop_test)

############################################
# 🧠 STEP 5: POST-HOC (PAIRWISE TESTS)
############################################

pairwise_test <- pairwise.prop.test(x, n, p.adjust.method = "bonferroni")

print(pairwise_test)

############################################
# 📊 STEP 6: REVENUE ANALYSIS (ANOVA)
############################################

anova_model <- aov(revenue ~ group, data = df)

summary(anova_model)

############################################
# 🔍 STEP 7: POST-HOC (TUKEY TEST)
############################################

tukey_result <- TukeyHSD(anova_model)

print(tukey_result)

############################################
# 📉 STEP 8: DECISION RULE
############################################

alpha <- 0.05

if (prop_test$p.value < alpha) {
  print("Significant difference in conversion rates across groups")
} else {
  print("No significant difference in conversion rates")
}

############################################
# 🧾 STEP 9: INTERPRETATION
############################################

cat("
INTERPRETATION:

1. Conversion rates increase from A → E, suggesting improvements in variants.

2. Proportion test checks if ANY group differs significantly.

3. Pairwise tests identify WHICH groups differ.

4. ANOVA checks revenue differences across all groups.

5. Tukey test shows which pairs differ significantly.

6. If Group E is significantly higher, it is the best candidate.

BUSINESS DECISION:
- If statistically significant → roll out best variant
- Else → collect more data or redesign experiment
")

