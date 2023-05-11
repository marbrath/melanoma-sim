library(RcppCNPy)
library(Rmpi)
library(numDeriv)


#try(detach("package:addsimR", unload=TRUE))
#detach("package:addsimR", unload=TRUE)

#detach("package:addsim", unload=TRUE)
#detach("package:addsimR", unload=TRUE)

#library(addsimR)

library(addsim)

# Prints names of packages that are loaded
print(loadedNamespaces()[match("addsim", loadedNamespaces())])
print(loadedNamespaces()[match("addsimR", loadedNamespaces())])

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
    #print(c('sickid', sick_id, 'ts', ts, 'bs', bs, 'gs', gs))

    if (is.nan(rhs) || rhs==0) {
         #print('rhs = 0 or is.nan(rhs)')
         #print(rhs)
         #print(lhs)

         return(-1e06)
    }

    value = (-1)**num_events * lhs * 1/rhs

    if (is.nan(value)) {
        print('in NaN')
        print(c(sick_id, 'lhs', lhs, 'rhs', rhs, 'num_events', num_events, 'ts', ts))

         return(-1e06)
    }

    res = log(value)

    return(res)
}

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("Usage: optimize.R <max children> <path/to/npy_files_xxx> <path/to/results/xxx")
}

max_children = as.integer(args[1])
root_path = args[2]
result_root_path = args[3]

all_sick_ids = npyLoad(file.path(root_path, 'sick_ids.npy'), "integer")
all_num_events = npyLoad(file.path(root_path, 'all_num_events.npy'), "integer")
all_ts = npyLoad(file.path(root_path, 'lifetimes.npy'), "integer")
all_rs = npyLoad(file.path(root_path, 'truncations.npy'), "integer")
all_bs = npyLoad(file.path(root_path, 'birthyears.npy'), "integer")
all_gs = npyLoad(file.path(root_path, 'genders.npy'), "integer")

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


mpi.bcast.Robj2slave(max_children)
mpi.bcast.Robj2slave(all_sick_ids)
mpi.bcast.Robj2slave(all_num_events)
mpi.bcast.Robj2slave(all_ts)
mpi.bcast.Robj2slave(all_rs)
mpi.bcast.Robj2slave(all_bs)
mpi.bcast.Robj2slave(all_gs)

mpi.bcast.Robj2slave(likelihood)
mpi.bcast.Robj2slave(l_term)

niter = 0

l_parallell = function(args){
    niter <<- niter + 1
    print(niter)

    var_e_ = exp(args[1])
    var_g_ = exp(args[2])
    k_ = args[3]
    beta_0_ = args[4]
    beta_1_ = args[5]
    beta_2_ = args[6]

    fam_size = 2 + max_children
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
                                            all_ts[(fam_size*(i-1)+1):(fam_size*i)],
                                            all_rs[(fam_size*(i-1)+1):(fam_size*i)],
                                            all_bs[(fam_size*(i-1)+1):(fam_size*i)],
                                            all_gs[(fam_size*(i-1)+1):(fam_size*i)],
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
    return(-l_parallell)
}

begin = proc.time()

init = c(
  #log(1.74), # switched var_g
  #log(0.51), # switched var_e
  log(0.51),
  log(1.74),
  4.32, # k
  -20, # beta_0
  0.27, # beta_1
  0.05 # beta_2
)
print("init, bounded, nlminb:")
print(init)
npySave(file.path(result_root_path, 'init.npy'), init)

lower_ = c(-20, -20, 0.1, -40, -10, -10)
upper_ = c(15, 10, 10, 25, 25, 25)

optim = nlminb(init, object= l_parallell, lower = lower_, upper = upper_, control=list(eval.max = 2000))
#optim = optim(init, l_parallell, method="BFGS", control = list(maxit=2000), hessian = TRUE)

end = proc.time()
print(end - begin)
print(optim)

if (optim$convergence == 0) {
  #npySave(file.path(result_root_path, 'optim.npy'), c(optim$par[2], optim$par[1], optim$par[3], optim$par[4], optim$par[5], optim$par[6]))
  npySave(file.path(result_root_path, 'optim.npy'), optim$par)
} else {
  npySave(file.path(result_root_path, 'optim.npy'), 0*optim$par)
}

#print("grad")
#G = grad(l_parallell, optim$par)
#print(G)
#npySave(file.path(result_root_path, 'jac.npy'), G)

#print("hessian")
hess = optimHess(optim$par, l_parallell)
#print(hess)


hess_inv = solve(hess)
#print(hess_inv)
npySave(file.path(result_root_path, 'hessian_inv'), hess_inv)


mpi.close.Rslaves()
mpi.quit()
