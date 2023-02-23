library(RcppCNPy)
library(Rmpi)
library(numDeriv)
library(addsim)

l_term = function(sick_id, num_events, ts, rs, bs, gs, var_e_, var_g_, k_, beta_0_, beta_1_, beta_2_) {

    lhs = likelihood(
      sick_id,
      ts,
      bs,
      gs,
      var_e_,
      var_g_,
      beta_0_,
      beta_1_,
      beta_2_,
      k_
    )

    rhs = likelihood(
      0,
      rs,
      bs,
      gs,
      var_e_,
      var_g_,
      beta_0_,
      beta_1_,
      beta_2_,
      k_
    )

    if (is.nan(rhs) || rhs==0) {
         #print('rhs = 0 or is.nan(rhs)')
         #print(rhs)
         #print(lhs)

         return(-1e06)
    }

    value = (-1)**num_events * lhs * 1/rhs

    if (is.nan(value)) {
        print('in NaN')
        print(c(lhs, rhs, num_events, ts))

         return(-1e06)
    }

    res = log(value)

    return(res)
}


all_sick_ids = npyLoad('npy_files/sick_ids.npy', "integer")
all_num_events = npyLoad('npy_files/all_num_events.npy', "integer")
all_ts = npyLoad('npy_files/lifetimes.npy', "integer")
all_rs = npyLoad('npy_files/truncations.npy', "integer")
all_bs = npyLoad('npy_files/birthyears.npy', "integer")
all_gs = npyLoad('npy_files/genders.npy', "integer")

indices = all_bs > 0
min_elem = min(all_bs[indices])
all_bs[indices] = (all_bs[indices] - min_elem)/10

if (!is.loaded("mpi_initialize")) {
    library("Rmpi")
}

nproc = strtoi(Sys.getenv("SLURM_NTASKS")) - 1

mpi.spawn.Rslaves(nslaves=nproc)

# In case R exits unexpectedly, have it automatically clean up
# resources taken up by Rmpi (slaves, memory, etc...)
.Last <- function(){
    if (is.loaded("mpi_initialize")){
        if (mpi.comm.size(1) > 0){
            print("Please use mpi.close.Rslaves() to close slaves.")
            mpi.close.Rslaves()
        }
        print("Please use mpi.quit() to quit R")
        .Call("mpi_finalize")
    }
}


mpi.bcast.Robj2slave(all_sick_ids)
mpi.bcast.Robj2slave(all_num_events)
mpi.bcast.Robj2slave(all_ts)
mpi.bcast.Robj2slave(all_rs)
mpi.bcast.Robj2slave(all_bs)
mpi.bcast.Robj2slave(all_gs)

mpi.bcast.Robj2slave(likelihood)
mpi.bcast.Robj2slave(l_term)

l_parallell = function(args){
    var_e_ = exp(args[1])
    var_g_ = exp(args[2])
    k_ = args[3]
    beta_0_ = args[4]
    beta_1_ = args[5]
    beta_2_ = args[6]

    n = length(all_sick_ids)

    l_parallell =
        sum(
            unlist(unname(
                mpi.parLapply(
                #lapply(
                    1:n,
                    FUN=function(i) l_term(
                                            all_sick_ids[i],
                                            all_num_events[i],
                                            all_ts[(10*(i-1)+1):(10*i)],
                                            all_rs[(10*(i-1)+1):(10*i)],
                                            all_bs[(10*(i-1)+1):(10*i)],
                                            all_gs[(10*(i-1)+1):(10*i)],
                                            var_e_,
                                            var_g_,
                                            k_,
                                            beta_0_,
                                            beta_1_,
                                            beta_2_
                                        )
                )

            ))
        )
        print("hei")
    return(-l_parallell)
}

begin = proc.time()

#init = c(log(7.02e2), log(0.32), log(4.34), 4.55, -0.03, 0.400)
#init = c(5, -2, -0.5, 4, -20)
#init = c(-3.097442, -3.926584, -3.951644, -28.553488, 0)
init = c(log(1/(81*exp(-2.75))), log(1/(81*exp(-3.94))), 3.619014, -28.551643, 0, 0)
#init = c(0.8*(-3.8977474), 0.95*43.4855928, 1.05*0.1483913,  0.95*2.1511297,  1.10*4.324515, 1.05*6)
#init = c(0.8*(-3.954447), 0.95*37.818952, 1.05*2.732414, 0.95*(-5.420834), 1.10*3.282227, 1.05*8.629212, 0)
print("init, bounded, nlminb:")
print(init)

lower_ = c(-20, -20, 0.1, -40, -10, -10)
upper_ = c(10, 10, 10, 25, 25, 25)

optim = nlminb(init, object= l_parallell, lower = lower_, upper = upper_, control=list(eval.max = 2000))
#optim = optim(init, l_parallell, method="BFGS", control = list(maxit=2000), hessian = TRUE)

end = proc.time()
print(end - begin)
print(optim)
print("grad")
print(grad(l_parallell, optim$par))

print("hessian")
#hess = optim$hessian
hess = optimHess(optim$par, l_parallell)
print(hess)
print("hessian inverse")
print(solve(hess))

mpi.close.Rslaves()
mpi.quit()
