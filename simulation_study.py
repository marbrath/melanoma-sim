import os
import subprocess
import sys


if len(sys.argv) != 4:
    print(f'USAGE: {sys.argv[0]} <num fams per year> <seed begin> <seed end>')

num_fams_per_year = int(sys.argv[1])
seed_begin = int(sys.argv[2])
seed_end = int(sys.argv[3])

if not os.path.exists('results'):
    os.mkdir('results')

for seed in range(seed_begin, seed_end):
    sys.stdout.write(f'Simulating data for seed {seed}...')
    sys.stdout.flush()
    subprocess.run(f'python3 sim/sim.py {seed} {num_fams_per_year}', shell=True, check=True)
    sys.stdout.write(f' done\n')
    sys.stdout.write(f'Optimizing for seed {seed}...')
    sys.stdout.flush()
    subprocess.run(f'mpirun -np 1 --use-hwthread-cpus Rscript optimize.R npy_files_{seed:04d} &> results/{seed:04d}.out', shell=True, check=True)
    sys.stdout.write(f' done\n')
