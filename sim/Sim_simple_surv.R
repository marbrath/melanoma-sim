library(survival)

##SPECIFY PARAMETERS:
#number of individuals:
n = 10000
#censor times:
t_c = rep(110*12,n)
#covariate l (birth year):
#l = rbinom(n,1,.4)
#covariate a (sex):
a = rbinom(n,1,0.5)
#Weibull scale parameter for baseline hazards:
lambda_01 = exp(-35.70+0.27*(1940-1853)/10)
#Weibull shape parameter for baseline hazards:
mu_01 = 4.32
#Regression coefficient for sex:
beta1_01 = 0.05
#Regression coefficient forbirth year:
#beta2_01 = 0.27
#Plot function curves:
curve(lambda_01 * exp(beta1_01*1) * mu_01 * x^(mu_01-1), from=1, to=max(t_c), xlab="t", ylab="h(t)", main="Conditional hazards", col=2)
curve(lambda_01 * exp(beta1_01*0) * mu_01 * x^(mu_01-1), from=1, to=max(t_c), xlab="t", ylab="h(t)", add=T,col=3)
legend(0.5,1,c("sex=1","sex=0"),lty=c(1,1),col=c("red","green"),cex=0.75)

if (FALSE) {
##SIMULATE n INDIVIDUAL TRAJECTORIES:
simdata = data.frame(id=integer(),event=integer(),tstart=integer(),tstop=integer(),a=integer())#,l=integer())
eventtime_vec = c()
for(i in 1:n){
      simdata.i = data.frame(id=i,event=1,tstart=0,tstop=t_c[i],a=a[i])#,l=l[i]) 
      #Cox-Weibull conditional hazard functions (using formulas from bender et al, 2005):
      U = runif(1,0,1)
      eventtime = (-(log(U))/(lambda_01)*exp(beta1_01*a[i]))^(1/mu_01)
      eventtime_vec = c(eventtime_vec,eventtime)
      if(simdata.i$tstop>=eventtime){
      	simdata.i$event=1
      	simdata.i$tstop = eventtime
      }
      simdata = rbind(simdata,simdata.i)
}
rownames(simdata) = 1:nrow(simdata)
#Save and print simulated data:
#save(simdata,file="simdata.Rdata")
head(simdata,20)

fit = survfit(Surv(tstop/12,event==1)~a, data=simdata, conf.type="plain")
plot(fit, mark.time=FALSE, xlab="Years",main="Kaplan-Meier",ylim=c(0.95,1))

dev.new()
hist(eventtime_vec/12)
}
