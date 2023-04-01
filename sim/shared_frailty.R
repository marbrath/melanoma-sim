library(survival)
library(frailtypack)
library(RcppCNPy)

#all_ts = npyLoad("npy_files_0000_old/lifetimes.npy", "integer")
#events = npyLoad("npy_files_0000_old/fam_events.npy", "integer")
#fam_ids = npyLoad("npy_files_0000_old/fam_ids.npy", "integer")
#all_gs = npyLoad("npy_files_0000_old/genders.npy", "integer")
all_ts = npyLoad("shared/lifetimes.npy")
events = npyLoad("shared/fam_events.npy", "integer")
fam_ids = npyLoad("shared/fam_ids.npy", "integer")
all_gs = npyLoad("shared/genders.npy", "integer")

events = events[all_ts > 0]
fam_ids = fam_ids[all_ts > 0]
all_gs = all_gs[all_ts > 0]
all_ts = all_ts[all_ts > 0]

set.seed(0)
df = data.frame(all_ts, events, all_gs, fam_ids)
df_short = df[df$fam_ids %in% sample(0:max(df$fam_ids),10000),]

fit = frailtyPenal(Surv(all_ts, events)~all_gs + cluster(fam_ids), hazard = "Weibull", RandDist = "Gamma", data=df_short) 
print(fit)

#Parameterne i rekkefølge: VAR_Z, beta_0, k, beta_1
param =c(fit$theta,log(fit$scale.weib[1]^(-fit$shape.weib[1])), fit$shape.weib[1], fit$coef[1])
print(param)

