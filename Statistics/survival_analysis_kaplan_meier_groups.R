library(survival)
data <- data.frame(
  time = c(2,3,6,6,7,10,15,15,16,27,
           4,5,8,9,12,14,18,20,22,25),
  
  status = c(1,0,1,1,1,0,1,1,1,1,
             1,1,0,1,0,1,0,1,1,0),
  
  treatment = c("A","A","A","A","A","A","A","A","A","A",
                "B","B","B","B","B","B","B","B","B","B"),
  
  age = c(65,70,72,68,66,74,69,71,67,73,
          60,62,59,63,61,64,58,66,65,67)
)

# Convert to factor
data$treatment <- as.factor(data$treatment)
km.model <- survfit(Surv(time, status) ~ treatment, data = data)

# Summary
km.model
summary(km.model)

plot(km.model,
     col = c("blue", "darkgreen"),
     lwd = 2,conf.int = TRUE,mark.time = TRUE,
     xlab = "Time (months)",ylab = "Survival Probability S(t)",
     main = "Kaplan-Meier Survival Curves (Drug A vs Drug B)",las = 1
)
legend("topright",legend = c("Drug A", "Drug B"),
       col = c("blue", "darkgreen"),
       lwd = 2,title = "Treatment",cex = 0.4)

abline(h = 0.5, col = "red", lty = 2)

survdiff(Surv(time, status) ~ treatment, data = data)
