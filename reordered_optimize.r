library(RcppCNPy)
library(Rmpi)
library(numDeriv)


library(addsim)

print(loadedNamespaces()[match("addsim", loadedNamespaces())])

l_term = function(sick_id, fam_events, num_events, ts, rs, bs, gs, var_e_, var_g_, k_, beta_0_, beta_1_, beta_2_) {
    n = length(ts)

    ts_ = c(
     ts[1:2][fam_events[1:2]],  # sick parents
     ts[1:2][!fam_events[1:2]], # non-sick parents
     ts[3:n][fam_events[3:n]],  # sick children
     ts[3:n][!fam_events[3:n]]  # non-sick children
    )
    rs_ = c(
     rs[1:2][fam_events[1:2]],  # sick parents
     rs[1:2][!fam_events[1:2]], # non-sick parents
     rs[3:n][fam_events[3:n]],  # sick children
     rs[3:n][!fam_events[3:n]]  # non-sick children
    )
    bs_ = c(
     bs[1:2][fam_events[1:2]],  # sick parents
     bs[1:2][!fam_events[1:2]], # non-sick parents
     bs[3:n][fam_events[3:n]],  # sick children
     bs[3:n][!fam_events[3:n]]  # non-sick children
    )
    gs_ = c(
     gs[1:2][fam_events[1:2]],  # sick parents
     gs[1:2][!fam_events[1:2]], # non-sick parents
     gs[3:n][fam_events[3:n]],  # sick children
     gs[3:n][!fam_events[3:n]]  # non-sick children
    )
    sick_id_ = (2**sum(fam_events[1:2]) - 1) + 4*(2**sum(fam_events[3:n]) - 1)

    lhs = likelihood(
      sick_id_,
      ts_,
      bs_,
      gs_,
      var_e_,
      var_g_,
      beta_0_,
      beta_1_,
      beta_2_,
      k_
    )

    rhs = likelihood(
      0,
      rs_,
      bs_,
      gs_,
      var_e_,
      var_g_,
      beta_0_,
      beta_1_,
      beta_2_,
      k_
    )

    if (is.nan(rhs) || rhs==0) {
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
all_fam_events = as.logical(npyLoad(file.path(root_path, 'all_fam_events.npy'), "integer"))
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
mpi.bcast.Robj2slave(all_fam_events)
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
                                            all_fam_events[(fam_size*(i-1)+1):(fam_size*i)],
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

end = proc.time()
print(end - begin)
print(optim)

if (optim$convergence == 0) {
  npySave(file.path(result_root_path, 'optim.npy'), optim$par)
} else {
  npySave(file.path(result_root_path, 'optim.npy'), 0*optim$par)
}

#hess = optimHess(optim$par, l_parallell)

#hess_inv = solve(hess)
#npySave(file.path(result_root_path, 'hessian_inv'), hess_inv)

mpi.close.Rslaves()
mpi.quit()
