
library(survival)
# Create the dataset
data <- data.frame(
  time = c(2, 3, 6, 6, 7, 10, 15, 15, 16, 27, 30, 32),
  died = c(1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1)
)
km.model <- survfit(Surv(time,died)~1,type='kaplan-meier')
km.model
summary(km.model)

plot(km.model,conf.int=T,
     xlab='Time(months)',ylab='%Alive=S(t)',
     main='KM-Model',las=1
)
plot(km.model,conf.int=T,
     xlab='Time(months)',ylab='%Alive=S(t)',
     main='KM-Model',las=1,mark.time=TRUE
)
abline(h=0.5,col="red")


