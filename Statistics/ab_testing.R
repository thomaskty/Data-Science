library(datarium)
library(dplyr)
library(ggplot2)

group <- c(
  rep("A", 50),
  rep("B", 50)
)

conversion <- c(
  # Group A (Control)
  0,0,1,0,0,1,0,0,1,0,
  0,1,0,0,0,1,0,0,1,0,
  0,0,1,0,0,1,0,0,0,1,
  0,0,1,0,0,1,0,0,0,1,
  0,0,1,0,0,1,0,0,1,0,
  
  # Group B (Treatment)
  1,0,1,1,0,1,1,0,1,1,
  0,1,1,0,1,1,0,1,1,0,
  1,1,0,1,1,0,1,1,0,1,
  1,0,1,1,0,1,1,0,1,1,
  0,1,1,0,1,1,0,1,1,0
)

df <- data.frame(group, conversion)

df_summary <- df %>%
  group_by(group) %>%
  summarise(
    total_users = n(),
    conversions = sum(conversion),
    conversion_rate = mean(conversion)
  )

ggplot(df_summary, aes(x = group, y = conversion_rate, fill = group)) +
  geom_bar(stat = "identity") +
  ylim(0,1) +
  theme_minimal() +
  labs(title = "Conversion Rate: A vs B",
       y = "Conversion Rate",
       x = "Group")

# Count conversions
conv_counts <- df %>%
  group_by(group) %>%
  summarise(conversions = sum(conversion),
            total = n())

# Extract values
x <- conv_counts$conversions
n <- conv_counts$total

# Proportion test
prop_test <- prop.test(x = x, n = n, alternative = "two.sided")
prop_test

prop_test_one <- prop.test(x = x, n = n, alternative = "greater")
prop_test_one

alpha <- 0.05
if (prop_test$p.value < alpha) {
  print("Reject H0: Significant difference in conversion rates")
} else {
  print("Fail to reject H0: No significant difference")
} # "Reject H0: Significant difference in conversion rates"

prop_test$conf.int # 0.95 

lift <- df_summary$conversion_rate[2] - df_summary$conversion_rate[1]
lift # 0.36 
