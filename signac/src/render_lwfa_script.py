"""
Render the Jinja2 template from templates/lwfa_script.j2 and produce lwfa_script.py.
"""
import jinja2
import pathlib


def get_template(template_file):
    """Get a jinja template."""
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(pathlib.Path("/").resolve()),
    )
    template = jinja_env.get_template(str(pathlib.Path(template_file).absolute()))
    return template


def write_lwfa_script(job):
    """Write lwfa_script.py in the job's workspace folder."""
    template = get_template("src/templates/lwfa_script.j2")
    rendered_template = template.render(sp=job.sp)

    p = pathlib.Path(job.ws) / "lwfa_script.py"
    p.write_text(rendered_template)


def main():
    """Main entry point."""
    import signac

    proj = signac.get_project(search=False)

    for job in proj:
        write_lwfa_script(job)


if __name__ == "__main__":
    main()
