from signac_dashboard import Dashboard
from signac_dashboard.modules import (
    StatepointList,
    DocumentList,
    ImageViewer,
    VideoViewer,
    FileList,
    Notes,
)
import unyt as u


def shave(number_as_str):
    before_dot, after_dot = number_as_str.split(".")
    if after_dot == "0":
        return before_dot
    else:
        return number_as_str


class MyDashboard(Dashboard):
    def job_sorter(self, job):
        return job.sp.a0, job.sp.zfoc

    def job_title(self, job):
        # ne = job.sp.n_e * (1 * u.meter ** (-3)).to(u.cm ** (-3))
        return f"a0 = {job.sp.a0}, zfoc = {job.sp.zfoc * 1e3}"


config = {
    "DASHBOARD_PATHS": ["."],
    "SECRET_KEY": b"\x99o\x90'/\rK\xf5\x10\xed\x8bC\xaa\x03\x9d\x99",
}

modules = [
    StatepointList(name="Params", enabled=False),
    DocumentList(name="Info", enabled=False),
    VideoViewer(name="Evolution", enabled=True),
    ImageViewer(name="Histogram", img_globs=["hist2d.png"], enabled=False),
    ImageViewer(name="Spectrum", img_globs=["final_histogram.png"], enabled=False),
    ImageViewer(name="All Plots", enabled=True),
    FileList(name="All Files", enabled=False),
    Notes(name="Notes", enabled=False),
]

dashboard = MyDashboard(config=config, modules=modules)

if __name__ == "__main__":
    dashboard.main()
