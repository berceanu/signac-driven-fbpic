#!/usr/bin/env python3
"""This module contains the operation functions for this project.

The workflow defined in this file can be executed from the command
line with

    $ python src/project.py run [job_id [job_id ...]]

See also: $ python src/project.py --help
"""
from flow import FlowProject, cmd, with_job
from signac import get_project
import os
import shutil
import random
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
import logging
logger = logging.getLogger(__name__)
import mypackage.code_api as code_api
import mypackage.util as util
import mypackage.talys_api as talys

# @with_job
logfname = 'rpa-project.log'

#####################
# UTILITY FUNCTIONS #
#####################

PNG_FILE = 'iso_all.png'

# https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file/18603065#18603065
def read_last_line(filename):
    with open(filename, "rb") as f:
        _ = f.readline()            # Read the first line.
        f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
        while f.read(1) != b"\n":   # Until EOL is found...
            f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
        last = f.readline()         # Read last line.
    return last

def isemptyfile(filename):
    return lambda job: job.isfile(filename) and os.stat(job.fn(filename)).st_size == 0

def file_contains(filename, text):
    """Checks if `filename` contains `text`."""
    return lambda job: job.isfile(filename) and text in open(job.fn(filename), 'r').read()

def arefiles(filenames):
    """Check if all `filenames` are in `job` folder."""
    return lambda job: all(job.isfile(fn) for fn in filenames)

###########################

code = code_api.NameMapping()

class Project(FlowProject):
    pass

####################
# OPERATION LABELS #
####################

@Project.label
def plotted(job):
    return job.isfile(PNG_FILE)

def _progress(job, temp, code_mapping=code_api.NameMapping()):
    fn = code_mapping.stdout_file(temp, state='excited')
    if job.isfile(fn):
        last_line = read_last_line(job.fn(fn)).decode('UTF-8')
        if 'pa1 = ' in last_line:
            percentage = last_line.split()[-1]
        else: # already finished the matrix calculation
            percentage = "100.00"
    else: # didn't yet start the matrix calculation
        percentage = "0.00"

    return f"run_{temp}_temp_excited_state: {float(percentage):.2f}%"

@Project.label
def progress_zero(job):
    return _progress(job, temp='zero', code_mapping=code)

@Project.label
def progress_finite(job):
    return _progress(job, temp='finite', code_mapping=code)


##########################
# GENERATING INPUT FILES #
##########################

def bin_files_exist(job, temp):
    """Check if job folder has the required .bin files for loading."""
    return arefiles(code.bin_files(temp))(job)

def _prepare_run(job, temp, code_mapping=code_api.NameMapping()):
    filter = {param:job.sp[param]
                for param in ('proton_number', 'neutron_number', 'angular_momentum', 'parity', 'temperature')}

    project = get_project()
    jobs_for_restart = [job for job in project.find_jobs(filter)
                            if bin_files_exist(job, temp)]
    assert job not in jobs_for_restart
    try:
        job_for_restart = random.choice(jobs_for_restart)
    except IndexError: # prepare full calculation
        logger.info('Full calculation required.')
        code_input = code_api.GenerateInputs(**job.sp,
                                                out_path=job.ws,
                                                mapping=code_mapping)
        code_input.write_param_files(temp)
        # job.doc[f'run_{temp}_temp_ground_state'] = True
    else:  # prepare restart
        logger.info('Restart possible.')
        code_input = code_api.GenerateInputs(**job.sp, 
                                                out_path=job.ws,                                                
                                                mapping=code_mapping,
                                                load_matrix=True)
        code_input.write_param_files(temp, state='excited')
        dotwelfn = code_mapping.wel_file(temp)
        shutil.copy(job_for_restart.fn(dotwelfn), job.fn(dotwelfn))
        for fn in code_mapping.bin_files(temp):
            shutil.copy(job_for_restart.fn(fn), job.fn(fn))
        job.doc['restarted_from'] = job_for_restart._id
        #job.doc[f'run_{temp}_temp_ground_state'] = False

@Project.operation
@Project.pre(lambda job: job.sp.temperature == 0)
@Project.post.isfile(code.input_file(temp='zero', state='excited'))
def prepare_run_zero(job):
    _prepare_run(job, temp='zero', code_mapping=code)

@Project.operation
@Project.pre(lambda job: job.sp.temperature != 0)
@Project.post.isfile(code.input_file(temp='finite', state='excited'))
def prepare_run_finite(job):
    _prepare_run(job, temp='finite', code_mapping=code)

#################
# RUNNING CODES #
#################

def _run_code(job, temp, state, codepath='../bin', code_mapping=code_api.NameMapping()):
    code = os.path.join(codepath, code_mapping.exec_file(temp, state))
    assert os.path.isfile(code), f"{code} not found!"

    stdout_file = job.fn(code_mapping.stdout_file(temp, state))
    stderr_file = job.fn(code_mapping.stderr_file(temp, state))

    run_command = f"{code} {job.ws} > {stdout_file} 2> {stderr_file}"
    command = f"echo {run_command} >> {logfname} ; {run_command}"

    return command


#### ZERO TEMP GROUND STATE ####
@Project.operation
@cmd
# @Project.pre.true('run_zero_temp_ground_state')
@Project.pre.isfile(code.input_file(temp='zero', state='ground')) 
@Project.post.isfile(code.wel_file(temp='zero')) 
@Project.post(file_contains(code.stderr_file(temp='zero', state='ground'),
                             'FINAL STOP'))
@Project.post(file_contains(code.stdout_file(temp='zero', state='ground'),
                             'Iteration converged'))
def run_zero_temp_ground_state(job):
    return _run_code(job, temp='zero', state='ground', code_mapping=code)


#### ZERO TEMP EXCITED STATE ####
@Project.operation
@cmd
@Project.pre.isfile(code.input_file(temp='zero', state='excited')) 
# @Project.pre.isfile(code.wel_file(temp='zero'))
# use all of run_zero_temp_ground_state's post-conditions as pre-conditions
@Project.pre.after(run_zero_temp_ground_state) 
@Project.post(arefiles(code.out_files(temp='zero')))
@Project.post(file_contains(code.stdout_file(temp='zero', state='excited'),
                             'program terminated without errors'))
@Project.post(isemptyfile(code.stderr_file(temp='zero', state='excited')))
def run_zero_temp_excited_state(job):
    return _run_code(job, temp='zero', state='excited', code_mapping=code)


#### FINITE TEMP GROUND STATE ####
@Project.operation
@cmd
# @Project.pre.true('run_finite_temp_ground_state')
@Project.pre.isfile(code.input_file(temp='finite', state='ground')) 
@Project.post.isfile(code.wel_file(temp='finite')) 
@Project.post(file_contains(code.stderr_file(temp='finite', state='ground'),
                             'FINAL STOP'))
@Project.post(file_contains(code.stdout_file(temp='finite', state='ground'),
                             'Iteration converged'))
def run_finite_temp_ground_state(job):
    return _run_code(job, temp='finite', state='ground', code_mapping=code)


#### FINITE TEMP EXCITED STATE ####
@Project.operation
@cmd
@Project.pre.isfile(code.input_file(temp='finite', state='excited')) 
# @Project.pre.isfile(code.wel_file(temp='finite'))
@Project.pre.after(run_finite_temp_ground_state)
@Project.post(arefiles(code.out_files(temp='finite')))
@Project.post(file_contains(code.stdout_file(temp='finite', state='excited'),
                             'program terminated without errors'))
@Project.post(isemptyfile(code.stderr_file(temp='finite', state='excited')))
def run_finite_temp_excited_state(job):
    return _run_code(job, temp='finite', state='excited', code_mapping=code)



############
# PLOTTING #
############

def _plot_iso(job, temp, code_mapping=code_api.NameMapping()):

    def _out_file_plot(job, ax, temp, skalvec, lorexc, code_mapping=code_api.NameMapping()):
    
        fn = job.fn(code_mapping.out_file(temp, skalvec, lorexc))
        
        df = pd.read_csv(fn, delim_whitespace=True, comment='#', skip_blank_lines=True,
                    header=None, names=['energy', 'transition_strength'])

        df = df[df.energy < 30] # MeV

        if lorexc == 'excitation':
            ax.vlines(df.energy, 0., df.transition_strength, colors='black')
        elif lorexc == 'lorentzian':
            ax.plot(df.energy, df.transition_strength, color='black')
        else:
            raise ValueError
        
        return df


    fig = Figure(figsize=(12, 6)) 
    canvas = FigureCanvas(fig)

    gs = GridSpec(2, 1)
    ax = {'isoscalar': fig.add_subplot(gs[0,0]),
        'isovector': fig.add_subplot(gs[1,0])}

    for skalvec in 'isoscalar', 'isovector':
        for sp in ("top", "bottom", "right"):
            ax[skalvec].spines[sp].set_visible(False)
        ax[skalvec].set(ylabel=r"$R \; (e^2fm^2/MeV)$")
        ax[skalvec].set_title(skalvec)
        for lorexc in 'excitation', 'lorentzian':
            df = _out_file_plot(job=job, ax=ax[skalvec], temp=temp, skalvec=skalvec, lorexc=lorexc, code_mapping=code_mapping)
            if lorexc == 'excitation' and job.sp.transition_energy != 0:
                df = df[np.isclose(df.energy, job.sp.transition_energy, atol=0.01)]
                ax[skalvec].vlines(df.energy, 0., df.transition_strength, colors='red')

    ax['isovector'].set(xlabel="E (MeV)")
    fig.subplots_adjust(hspace=0.3)

    element, mass = util.split_element_mass(job)
    fig.suptitle(fr"Transition strength distribution of ${{}}^{{{mass}}} {element} \; {job.sp.angular_momentum}^{{{job.sp.parity}}}$ at T = {job.sp.temperature} MeV")

    canvas.print_png(job.fn(PNG_FILE))


@Project.operation
@Project.pre(arefiles(code.out_files(temp='zero')))
@Project.post.isfile(PNG_FILE)
def plot_zero(job):
    _plot_iso(job, temp='zero', code_mapping=code)

@Project.operation
@Project.pre(arefiles(code.out_files(temp='finite')))
@Project.post.isfile(PNG_FILE)
def plot_finite(job):
    _plot_iso(job, temp='finite', code_mapping=code)

####################################
# EXTRACT DIPOLE TRANSITIONS TABLE #
####################################

def _extract_transitions(job, temp, code_mapping=code_api.NameMapping()):
    first_marker = "1=n/2=p       E/hole      E/particle  XX-YY/%"
    last_marker = "Sum XX-YY after normalization *"
    #
    infn = job.fn(code_mapping.stdout_file(temp, state='excited'))
    outfn = job.fn('dipole_transitions.txt')
    #
    with open(infn, 'r') as infile, open(outfn, 'w') as outfile:
        copy = False
        for line in infile:
            if first_marker in line.strip():
                copy = True
            elif last_marker in line.strip():
                copy = False
            elif copy:
                outfile.write(line)

@Project.operation
@Project.pre(file_contains(code.stdout_file(temp='zero', state='excited'),
                             "1=n/2=p       E/hole      E/particle  XX-YY/%"))
@Project.post.isfile('dipole_transitions.txt')
def dipole_trans_zero(job):
    _extract_transitions(job, temp='zero', code_mapping=code)

@Project.operation
@Project.pre(file_contains(code.stdout_file(temp='finite', state='excited'),
                             "1=n/2=p       E/hole      E/particle  XX-YY/%"))
@Project.post.isfile('dipole_transitions.txt')
def dipole_trans_finite(job):
    _extract_transitions(job, temp='finite', code_mapping=code)


#################################
# GENERATE INPUT FOR TALYS CODE #
#################################

def z_fn(job):
    return 'z{:03d}'.format(job.sp.proton_number)

def talys_template_file(job, top_level_dir='src/templates', fname=None):
    if not fname:
        fname = z_fn(job)
    full_path = os.path.join(top_level_dir, fname)
    return full_path


def _generate_talys_input(job, temp, code_mapping=code_api.NameMapping()):
    job_mass_number = job.sp.proton_number + job.sp.neutron_number
    fn = job.fn(code_mapping.out_file(temp=temp, 
                            skalvec='isovector', lorexc='lorentzian'))

    lorvec_df = talys.lorvec_to_df(fname=fn, 
                        Z=job.sp.proton_number, A=job_mass_number)

    talys_dict = talys.fn_to_dict(fname=talys_template_file(job))
    talys_df = talys.dict_to_df(talys_dict)

    mass_numbers = talys.atomic_mass_numbers(talys_df)
    logger.info("{} contains atomic mass numbers from A={} to A={}.".format(
        talys_template_file(job),
        mass_numbers.min(), mass_numbers.max()))


    if job_mass_number in mass_numbers:
        talys_df_new = talys.replace_table(Z=job.sp.proton_number,
                                           A=job_mass_number,
                                talys=talys_df, lorvec=lorvec_df)
        new_talys_dict = talys.df_to_dict(talys_df_new)
        talys.dict_to_fn(new_talys_dict, fname=job.fn(z_fn(job)))
        job.doc['talys_input'] = z_fn(job)
    else:
        logger.warning("(Z,A)=({},{}) not found in {}!".format(
            job.sp.proton_number, job_mass_number,
            talys_template_file(job)))


@Project.operation
@Project.pre(lambda job: os.path.isfile(talys_template_file(job)))
@Project.pre.isfile(code.out_file(temp='zero', 
                            skalvec='isovector', lorexc='lorentzian'))
@Project.post(lambda job: job.isfile(z_fn(job)))
def generate_talys_input_zero(job):
    _generate_talys_input(job, temp='zero', code_mapping=code)

@Project.operation
@Project.pre(lambda job: os.path.isfile(talys_template_file(job)))
@Project.pre.isfile(code.out_file(temp='finite', 
                            skalvec='isovector', lorexc='lorentzian'))
@Project.post(lambda job: job.isfile(z_fn(job)))
def generate_talys_input_finite(job):
    _generate_talys_input(job, temp='finite', code_mapping=code)



if __name__ == '__main__':
    logging.basicConfig(
        filename=logfname,
        format='%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger.info('==RUN STARTED==')
    Project().main()
    logger.info('==RUN FINISHED==')
