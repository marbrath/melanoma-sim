import numpy as np
import os
import sys

if len(sys.argv) < 2:
    print(f'USAGE: {sys.argv[0]} <path to results root> [max children]')
    sys.exit(1)

result_root = sys.argv[1]
max_children = int(sys.argv[2]) if len(sys.argv) > 2 else None
all_optim = []
all_hess_inv = []

result_suffix = f'_{max_children:02d}' if max_children is not None else None

for seedname in next(os.walk(result_root))[1]:
    if max_children is not None and seedname[-len(result_suffix):] != result_suffix:
        continue

    result_path = os.path.join(result_root, seedname)

    try:
        res = np.load(os.path.join(result_path, 'optim.npy'))
        all_optim.append(res)
    except:
        print(f'{seedname} missing')
        continue

    try:
        hess_inv = np.load(os.path.join(result_path, 'hessian_inv'))
        all_hess_inv.append(hess_inv.diagonal())
    except:
        all_hess_inv.append(np.zeros(6))
        print(f'{seedname} hessinv missing')


print(len(all_optim))
np.save(os.path.join(result_root, 'all_optim.npy'), np.stack(all_optim))
np.save(os.path.join(result_root, 'all_hess_inv.npy'), np.stack(all_hess_inv))
