def two_parameters_study(project, keys=("a0", "n_e")):
    spectra = list()

    vy = job_util.get_key_values(project, keys[0])
    vx = job_util.get_key_values(project, keys[1])

    for val_y, val_x in product(vy, vx):
        job = next(iter(project.find_jobs(filter={keys[0]: val_y, keys[1]: val_x})))
        spectrum = construct_electron_spectrum(job)
        spectra.append(spectrum)


def main():
    pass


if __name__ == '__main__':
    main()
