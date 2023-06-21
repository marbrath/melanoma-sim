import numpy as np
import sys
import os
import shutil


def truncate_sim(seed, in_max_children, out_max_children):
    src_path = 'sim-output/npy_files_%04d_%02d' % (seed, in_max_children)
    root_path = 'sim-output/npy_files_%04d_%02d' % (seed, out_max_children)

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    shutil.copy(os.path.join(src_path, 'fam_ids.npy'), os.path.join(root_path, 'fam_ids.npy'))

    fam_events = np.load(src_path + '/fam_events.npy')

    target_shape = fam_events.shape
    fam_genders = np.load(src_path + '/genders.npy').reshape(target_shape)
    fam_birthyears = np.load(src_path + '/birthyears.npy').reshape(target_shape)
    fam_lifetimes = np.load(src_path + '/lifetimes.npy').reshape(target_shape)

    out_fam_size = 2 + out_max_children

    fam_genders = fam_genders[:, :out_fam_size]
    fam_events = fam_events[:, :out_fam_size]
    fam_birthyears = fam_birthyears[:, :out_fam_size]
    fam_lifetimes = fam_lifetimes[:, :out_fam_size]

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
    if len(sys.argv) != 4:
        print('Usage: truncate_sim.py <seed> <original max children> <truncate to max children>')
        sys.exit(1)

    seed = int(sys.argv[1])
    in_max_children = int(sys.argv[2])
    out_max_children = int(sys.argv[3])
    truncate_sim(seed, in_max_children, out_max_children)
