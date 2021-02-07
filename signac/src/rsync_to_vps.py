"""Updates the signac workspace and dashboard.py on a VPS instance."""
import pathlib
import signac
from util import shell_run
import shutil


def main():
    """Main entry point."""
    project = signac.get_project(search=False)
    scratch_pr_path = pathlib.Path(project.workspace())

    # copy the workspace to current dir, excluding large files
    rsync_scratch_to_here = rf"rsync -amvP --exclude='*.h5' --exclude='*.txt' --exclude='*.npz' --exclude='*/rhos' --exclude='*/phasespaces' {scratch_pr_path}/ workspace"
    print(rsync_scratch_to_here)
    shell_run(rsync_scratch_to_here, shell=True)

    p = pathlib.Path.cwd()
    rc = p / "signac.rc"
    bck = p / "signac.rc.bck"

    # copy `signac.rc` to `signac.rc.bck`
    contents = rc.read_text()
    bck.write_text(contents)

    # change `workspace_dir` inside `signac.rc`
    lines = contents.splitlines()
    before_eq, _ = lines[1].split("=")
    after_eq = " workspace"
    lines[1] = "=".join((before_eq, after_eq))
    new_contents = "\n".join(lines)
    rc.write_text(new_contents)

    # remove workspace on VPS via SSH
    rm_cmd = rf"ssh signac-dashboard-ubuntu-4gb 'rm -rf ~/Development/signac'"
    print(rm_cmd)
    shell_run(rm_cmd, shell=True)

    # copy local workspace/ via SSH
    rsync_here_to_vps = rf"rsync -amvP --include='signac.rc' --include='dashboard.py' --include='wsgi.py' --include='*/' --include='workspace/***' --exclude='*' ./ signac-dashboard-ubuntu-4gb:Development/signac"
    print(rsync_here_to_vps)
    shell_run(rsync_here_to_vps, shell=True)

    # copy `signac.rc.bck` to `signac.rc`
    contents = bck.read_text()
    rc.write_text(contents)
    bck.unlink()

    # delete local workspace/
    p = pathlib.Path.cwd() / "workspace/"
    shutil.rmtree(p)

    # restart VPS
    reboot_cmd = rf"ssh -t signac-dashboard-ubuntu-4gb 'sudo /sbin/reboot'"
    print(reboot_cmd)
    shell_run(reboot_cmd, shell=True)


if __name__ == "__main__":
    main()
