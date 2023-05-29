import subprocess
import sys
import time


if len(sys.argv) != 4:
    print(f'USAGE: {sys.argv[0]} <max children> <seed begin> <seed end>')
    sys.exit(1)


max_children = int(sys.argv[1])
seed_begin = int(sys.argv[2])
seed_end = int(sys.argv[3])


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
    sys.stdout.write(f'Shuffling data for seed {seed}...')
    sys.stdout.flush()
    begin = time.monotonic()
    subprocess.run(f'python3 sim/shuffle_sim.py {seed} {max_children}', shell=True, check=True)
    end = time.monotonic()
    sys.stdout.write(f' done (took {get_time_str(end - begin)})\n')
    sys.stdout.flush()
