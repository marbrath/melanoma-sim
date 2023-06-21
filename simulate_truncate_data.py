import subprocess
import sys
import time


if len(sys.argv) != 5:
    print(f'USAGE: {sys.argv[0]} <original max children> <truncate max children> <seed begin> <seed end>')
    sys.exit(1)


in_max_children = int(sys.argv[1])
out_max_children = int(sys.argv[2])
seed_begin = int(sys.argv[3])
seed_end = int(sys.argv[4])


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
    sys.stdout.write(f'Truncating data for seed {seed}...')
    sys.stdout.flush()
    begin = time.monotonic()
    subprocess.run(f'python3 sim/truncate_sim.py {seed} {in_max_children} {out_max_children}', shell=True, check=True)
    end = time.monotonic()
    sys.stdout.write(f' done (took {get_time_str(end - begin)})\n')
    sys.stdout.flush()
