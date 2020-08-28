"""
Merge distributed analysis sets from a FileHandler.

Usage:
    merge.py <base_path> [--cleanup=<tf>]

Options:
    --cleanup=<tf>   # Delete distributed files after merging [default: True]

"""

if __name__ == "__main__":

    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from mpi4py import MPI

    args = docopt(__doc__)
    base_path = args['<base_path>']
    clean_up = bool(args['--cleanup'])
    post.merge_analysis(base_path, cleanup=clean_up)
    # Merge snapshots into one file (might make a really big file)
    merge_to_one = True
    if merge_to_one == True:
        import pathlib
        set_paths = list(pathlib.Path(base_path).glob('*.h5'))
        post.merge_sets(base_path+'/consolidated_analysis.h5', set_paths, cleanup=clean_up)
