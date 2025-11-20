import numpy as np

'''
- map_n_downsample: maps the values of n to the downsample factor. It can be 
  increased, to obtain smoother figures, reduce the size of the output files, 
  and accelerate the production of figures.

- map_n_procs: maps the values of n to the number of tasks used in the sbatch 
  script. If the number of tasks has been changed, this has to change 
  accordingly.

- instances_per_shape: the number of instances randomly sampled for each shape.
  Must match the number used in the sbatch scripts.

- values_instance: we have 4 ratios over optimal per instance, one for each of 
  the following sets: a fanning-out set, fanning-out set plus 1 added variant 
  through expansion, fanning-out set plus 2 added variants through expansion, 
  the left-to-right evaluation.
'''

if __name__ == "__main__":
    list_n = [5, 6, 7]
    map_n_downsample = {5: 10, 6: 100, 7: 1000} # downsample factor depends on n
    map_n_nprocs = {5: 100, 6: 100, 7: 1000} #
    instances_per_shape = 1000
    values_instance = 4

    base_fname_out = "../results/experiment1/trimmed_instances_n{length}.npz"

    for n in list_n:
        nprocs = map_n_nprocs[n]
        n_shapes = 10**n - 9**n
        total_instances = n_shapes * instances_per_shape
        DF = map_n_downsample[n]
        array = np.empty((total_instances, values_instance), dtype=np.float32)
        base_path = f"../results/experiment1/n{n}/instances_proc_"
        filled_rows = 0

        for i in range(0, nprocs):
            proc_filename = base_path + f"{i}.bin"
            with open(proc_filename, 'rb') as f:
                new_chunk = np.fromfile(f, dtype=np.float32)
                newshape = (len(new_chunk) // values_instance, values_instance)
                new_chunk = np.reshape(new_chunk, newshape, order='F')

            lines_to_fill = newshape[0]
            array[filled_rows:filled_rows+lines_to_fill] = new_chunk
            filled_rows += lines_to_fill
        
        array.sort(axis=0)
        trimmed = array[::DF, :].copy()
        np.savez_compressed(base_fname_out.format(length=n), a=trimmed)
        del array