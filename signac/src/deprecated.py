import multiprocessing
from glob import glob

from flow import FlowProject, directives


class Project(FlowProject):
    pass


def generate_png_files():
    pass


@Project.operation
@directives(nproc=8)  # only needed when submitting
def generate_movie(job):
    h5_files = glob(job.fn('diags/hdf5/*.h5'))
    with multiprocessing.Pool(nproc) as pool:
        pool.map(generate_png_files, h5_files)
    ffmpeg.input(job.fn(r'%016d.png'), framerate=24).output(job.fn('movie.mp4').run()

# see also:
# https://docs.signac.io/projects/core/en/latest/api.html#the-h5storemanager
