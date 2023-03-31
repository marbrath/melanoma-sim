import numpy as np
import os
import sys

if len(sys.argv) < 2:
    print(f'USAGE: {sys.argv[0]} <path to results root>')
    sys.exit(1)

result_root = sys.argv[1]
all_optim = []

for seedname in next(os.walk(result_root))[1]:
    result_path = os.path.join(result_root, seedname)
    all_optim.append(np.load(os.path.join(result_path, 'optim.npy')))

np.save(os.path.join(result_root, 'all_optim.npy'), np.stack(all_optim))
