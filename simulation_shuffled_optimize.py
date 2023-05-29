import os
import subprocess
import sys
import time


if len(sys.argv) != 4:
    print(f'USAGE: {sys.argv[0]} <max children> <seed begin> <seed end>')
    sys.exit(1)

max_children = int(sys.argv[1])
seed_begin = int(sys.argv[2])
seed_end = int(sys.argv[3])

results_path = 'sim-output/shuffled_results'
if not os.path.exists(results_path):
    os.mkdir(results_path)


def get_time_str(duration):
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60

    if hours > 0:
        return f'{hours}h {minutes}m {int(seconds)}s'

    if minutes > 0:
        return f'{minutes}m {int(seconds)}s'

    return f'{seconds:.2f}s'


for seed in range(seed_begin, seed_end):
    seed_results_path = os.path.join(results_path, f'{seed:04d}_{max_children:02d}')

    if not os.path.exists(seed_results_path):
        os.mkdir(seed_results_path)

    command = f'mpirun -np 1 --use-hwthread-cpus Rscript optimize.r {max_children} sim-output/shuffle_npy_files_{seed:04d}_{max_children:02d} {seed_results_path} &> {seed_results_path}/log.out'

    sys.stdout.write(f'Optimizing for seed {seed}...')
    sys.stdout.flush()

    try:
        begin = time.monotonic()
        subprocess.run(command, shell=True, check=True)
        end = time.monotonic()
        sys.stdout.write(f' done (took {get_time_str(end - begin)})\n')
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        sys.stdout.write(f' failed\n')
        sys.stdout.flush()
