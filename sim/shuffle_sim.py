import numpy as np
import sys
import os
import shutil


def shuffle_sim(seed, max_children):
    np.random.seed(seed)

    src_path = 'sim-output/npy_files_%04d_%02d' % (seed, max_children)
    root_path = 'sim-output/shuffle_npy_files_%04d_%02d' % (seed, max_children)

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    shutil.copy(os.path.join(src_path, 'fam_num_children.npy'), os.path.join(root_path, 'fam_num_children.npy'))
    shutil.copy(os.path.join(src_path, 'fam_ids.npy'), os.path.join(root_path, 'fam_ids.npy'))
    shutil.copy(os.path.join(src_path, 'all_num_events.npy'), os.path.join(root_path, 'all_num_events.npy'))

    fam_events = np.load(src_path + '/fam_events.npy')

    target_shape = fam_events.shape
    num_fams, fam_size = target_shape
    fam_genders = np.load(src_path + '/genders.npy').reshape(target_shape)
    fam_birthyears = np.load(src_path + '/birthyears.npy').reshape(target_shape)
    fam_lifetimes = np.load(src_path + '/lifetimes.npy').reshape(target_shape)
    fam_truncation_times = np.load(src_path + '/truncations.npy').reshape(target_shape)


    fam_order = np.empty(target_shape, dtype=fam_genders.dtype)
    fam_order[:, :] = np.arange(target_shape[1])[None]

    np.random.default_rng().permuted(fam_order[:, :2], out=fam_order[:, :2], axis=1)
    np.random.default_rng().permuted(fam_order[:, 2:], out=fam_order[:, 2:], axis=1)

    fam_idx = np.arange(num_fams)[:, None]
    has_value = fam_birthyears[fam_idx, fam_order] > 0
    fam_order[:, 2:] = fam_order[:, 2:][fam_idx, (~has_value[:, 2:]).argsort(axis=1)] # this packs children up front

    fam_genders = fam_genders[fam_idx, fam_order]
    fam_events = fam_events[fam_idx, fam_order]
    fam_birthyears = fam_birthyears[fam_idx, fam_order]
    fam_lifetimes = fam_lifetimes[fam_idx, fam_order]
    fam_truncation_times = fam_truncation_times[fam_idx, fam_order]

    fam_genders = fam_genders.ravel().astype('int64')
    fam_birthyears = fam_birthyears.ravel().astype('int64')
    fam_lifetimes = fam_lifetimes.ravel().astype('int64')
    event_bits = np.packbits(fam_events, bitorder='little', axis=1).astype('int64')
    assert(event_bits.shape[1] <= 3) # we here assume that family_size <= 24, which it is not

    fam_sick_ids = event_bits[:, 0]

    if event_bits.shape[1] >= 2:
        fam_sick_ids += (event_bits[:, 1] << 8)

    if event_bits.shape[1] >= 3:
        fam_sick_ids += (event_bits[:, 2] << 16)

    fam_truncation_times = (fam_lifetimes.ravel()*0).astype('int64')

    all_fam_events = np.vstack(fam_events).ravel().astype('int64')

    np.save(root_path + '/fam_events', fam_events)
    np.save(root_path + '/all_fam_events', all_fam_events)
    np.save(root_path + '/genders', fam_genders)
    np.save(root_path + '/birthyears', fam_birthyears)
    np.save(root_path + '/lifetimes', fam_lifetimes)
    np.save(root_path + '/sick_ids', fam_sick_ids)
    np.save(root_path + '/truncations', fam_truncation_times) 


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: shuffle_sim.py <seed> <max children>')
        sys.exit(1)

    seed = int(sys.argv[1])
    max_children = int(sys.argv[2])
    shuffle_sim(seed, max_children)
