# faithful contains measurements from the geyser.
# eruptions : Eruption time in mins
# waiting : Waiting time to next eruption (in mins)
data(faithful)

waiting_times <- faithful$waiting
summary(waiting_times)

mean_waiting_time <- mean(waiting_times) # ≈ 70.9 minutes 
lambda_hat <- 1 / mean(waiting_times) # ≈ 0.0141 events per minute

hist(waiting_times,
     breaks = 30,
     probability = TRUE,
     main = "Waiting Times (Old Faithful Geyser)",
     xlab = "Time (minutes)")


# P(T <= 70)
# The probability that the waiting time until the next eruption is at most 70 minutes
# After an eruption occurs, what is the probability that the next eruption happens within 70 minutes?
p1 <- pexp(70, rate = lambda_hat) # 0.62 
p2 <- 1 - pexp(70, rate = lambda_hat) # 0.37 
p3 <- pexp(90, rate = lambda_hat) - pexp(60, rate = lambda_hat) # 0.148 

# corresponding empirical probabilities 
emp_p1 <- sum(waiting_times <= 70) / length(waiting_times) # 0.39
emp_p2 <- sum(waiting_times>70)/length(waiting_times) # 0.60 
emp_p3 <- sum(waiting_times > 60 & waiting_times <= 90) / length(waiting_times)  # 0.67 


# Memoryless property check
s <- 60
t <- 10
lhs <- sum(waiting_times > (s + t)) / sum(waiting_times > s) # 0.873 
rhs <- sum(waiting_times > t) / length(waiting_times) # 1 

# Empirical Mean and variance
mean(waiting_times) # 70.89706
var(waiting_times) # 184.8233

# theoretical (exponential)
theoretical_mean <- 1 / lambda_hat  # 70.89706
theorectical_var <- 1 / (lambda_hat^2)  # 5026.393


# -----------------------------------------
# Comparing Exponential Distributions
# -----------------------------------------
x <- seq(0, 10, length.out = 1000)
lambda1 <- 0.5
lambda2 <- 1
lambda3 <- 2

# Plot first curve
plot(x, dexp(x, rate = lambda1),
     type = "l",lwd = 2,col = "blue",ylim = c(0, 2.5),
     xlab = "Time",ylab = "Density",
     main = "Exponential Distribution for Different Rates")

lines(x, dexp(x, rate = lambda2),col = "red",lwd = 2)
lines(x, dexp(x, rate = lambda3),col = "darkgreen",lwd = 2)

# Add legend
legend("topright",
       legend = c("lambda = 0.5", "lambda = 1", "lambda = 2"),
       col = c("blue", "red", "darkgreen"),
       lwd = 2,bty = "n")

# cdf comparison 
plot(x, pexp(x, rate = lambda1),
     type = "l", lwd = 2, col = "blue",
     xlab = "Time", ylab = "CDF",
     main = "CDF Comparison")

lines(x, pexp(x, rate = lambda2), col = "red", lwd = 2)
lines(x, pexp(x, rate = lambda3), col = "darkgreen", lwd = 2)

legend("bottomright",
       legend = c("lambda = 0.5", "lambda = 1", "lambda = 2"),
       col = c("blue", "red", "darkgreen"),
       lwd = 2,
       bty = "n")