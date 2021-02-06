"""Updates the signac workspace and dashboard.py on a VPS instance."""
import pathlib
import signac
from util import shell_run


def main():
    """Main entry point."""
    project = signac.get_project(search=False)
    scratch_pr_path = pathlib.Path(project.workspace())

    rsync_scratch_to_here = rf"rsync -amvP --exclude='*.h5' --exclude='*.txt' --exclude='*.npz' --exclude='*/rhos' --exclude='*/phasespaces' {scratch_pr_path}/ workspace"
    print(rsync_scratch_to_here)
    shell_run(rsync_scratch_to_here, shell=True)

    rsync_here_to_vps = rf"rsync -amvP --include='dashboard.py' --include='wsgi.py' --include='*/' --include='workspace/***' --exclude='*' ./ signac-dashboard-ubuntu-4gb:Development/signac"
    print(rsync_here_to_vps)
    shell_run(rsync_here_to_vps, shell=True)


if __name__ == "__main__":
    main()
