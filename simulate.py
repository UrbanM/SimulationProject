# This script makes simulations of auditory evoked fields.
# Copyright (C) 2021  Urban Marhl; email: urban.marhl@imfm.si
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import mne
import megtools.vector_functions as vfun
import megtools.pyread_biosig as pbio
import random


def main():
    # simulate_rms("test.obj", system_glob="OPM", magnetometer_type=1, components=["rad", "tan", "ver"])
    # simulate("test-all.obj", system_glob="OPM", magnetometer_type=1, components=["rad", "tan", "ver"])
    # simulate("test.obj", system_glob="SQUID", magnetometer_type=6, n_jobs=1)
    # simulate("simulate2508-squid1.obj", system_glob="SQUID", magnetometer_type=1, n_jobs=1)
    simulate("simulate2708-all.obj", system_glob="OPM", magnetometer_type=1, components=["rad", "tan", "ver"], n_jobs=1)

    # test("simulate3007-rad-bak.obj")

    # plot_contour("simulate3007-all-bak.obj", fname1="simulate3007-rad-bak.obj", fname2="simulate3007-tan-bak.obj",
    #             fname3="simulate3007-ver-bak.obj", labels=["all", "rad", "tan", "ver"])
    # visualize("simulate2707-all.obj", fname1="simulate2707-rad.obj", fname2="simulate2707-tan.obj",
    #           fname3="simulate2707-ver.obj", labels=["all", "rad", "tan", "ver"])

    # visualize("simulate2707-all.obj", fname1="simulate2707-rad.obj", fname2="simulate2707-tan.obj",
    #           fname3="simulate2707-ver.obj", labels=["all", "rad", "tan", "ver"])

    # visualize3("test.obj")

    # visualize("simulate1608,1908-squid1.obj", fname1="simulate1608,1908-squid2.obj", fname2="simulate1608,1908-squid3.obj",
    #           fname3="simulate1608,1908-squid4.obj", fname4="simulate1608,1908-squid5.obj",
    #           fname5="simulate1608,1908-squid6.obj", labels=["axial gradiometer", "magnetometer (rad.)",
    #           "planar gradiometer (lat.)", "planar gradiometer (long.)", "magnetometer (tan. lat.)",
    #                                                          "magnetometer (tan. long.)"])

    # visualize3("simulate2508-squid1.obj", fname1="simulate2508-squid2.obj", fname2="simulate2508-squid3.obj",
    #           fname3="simulate2508-squid4.obj", fname4="simulate2508-squid5.obj",
    #           fname5="simulate2508-squid6.obj", labels=["axial gradiometer", "magnetometer (rad.)",
    #           "planar gradiometer (lat.)", "planar gradiometer (long.)", "magnetometer (tan. lat.)",
    #                                                          "magnetometer (tan. long.)"])

    # visualize2("simulate1708-squid1.obj", fname1="simulate1708-squid2.obj", fname2="simulate1708-squid3.obj",
    #           fname3="simulate1708-squid4.obj", fname4="simulate1708-squid5.obj",
    #           fname5="simulate1708-squid6.obj", labels=["axial gradiometer", "magnetometer (rad.)",
    #           "planar gradiometer (lat.)", "planar gradiometer (long.)", "magnetometer (tan. lat.)",
    #                                                          "magnetometer (tan. long.)"])

    # visualize_relative("simulate2508-all-relative.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #           fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"])

    # visualize_relative("simulate2508-all-relative.obj", labels=["all"])
    # test("simulate3007,1908-all.obj")


    #
    # merge_two("simulate1608-squid1.obj", "simulate1908-squid1.obj", "simulate1608,1908-squid1.obj")
    # merge_two("simulate1608-squid2.obj", "simulate1908-squid2.obj", "simulate1608,1908-squid2.obj")
    # merge_two("simulate1608-squid3.obj", "simulate1908-squid3.obj", "simulate1608,1908-squid3.obj")
    # merge_two("simulate1608-squid4.obj", "simulate1908-squid4.obj", "simulate1608,1908-squid4.obj")
    # merge_two("simulate1608-squid5.obj", "simulate1908-squid5.obj", "simulate1608,1908-squid5.obj")
    # merge_two("simulate1608-squid6.obj", "simulate1908-squid6.obj", "simulate1608,1908-squid6.obj")

    # merge_two("simulate3007-all.obj", "simulate1908-all.obj", "simulate3007,1908-all.obj")
    # merge_two("simulate3007-rad.obj", "simulate1908-rad.obj", "simulate3007,1908-rad.obj")
    # merge_two("simulate3007-tan.obj", "simulate1908-tan.obj", "simulate3007,1908-tan.obj")
    # merge_two("simulate3007-ver.obj", "simulate1908-ver.obj", "simulate3007,1908-ver.obj")

    # plot_contour("simulate3007,1908-all.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #             fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"], parameter="snr")

    # visualize4("simulate3007,1908-all.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #            fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"])


    # visualize("test.obj", labels=["ver"])

    return


def visualize_relative(fname, fname1=None, fname2=None, fname3=None, labels=["empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.calc_avgs()
        obj1.calc_stds()

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.calc_avgs()
        obj2.calc_stds()

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.calc_avgs()
        obj3.calc_stds()

    # xscale = 10 ** 15
    xscale = 1
    # xname = "$\sigma \mathrm{[fT]}$"
    # xname = "$ SNR   \mathrm{[dB]}$"
    xname = "ratio"

    fig1, ax1 = plot_one_var(obj.noisestr[0], obj.avg_avgcc, yerr=obj.std_avgcc, xname=xname, yname="$\mathrm{CC}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig2, ax2 = plot_one_var(obj.noisestr[0], obj.avg_avgre, yerr=obj.std_avgre, xname=xname, yname="$\mathrm{RE}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig3, ax3 = plot_one_var(obj.noisestr[0], obj.avg_avgdist, yerr=obj.std_avgdist, xname=xname,
                             yname="$d(p1, p2) \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=True)
    fig4, ax4 = plot_one_var(obj.noisestr[0], obj.avg_snr, yerr=obj.std_snr, xname=xname, yname="$\mathrm{SNR}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        res = [idx for idx, val in enumerate(obj.spontnoise[0]) if val == 0.0]

        add_one_var_fig(fig1, ax1, obj1.noisestr[0], obj1.avg_avgcc, yerr=obj1.std_avgcc, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig2, ax2, obj1.noisestr[0], obj1.avg_avgre, yerr=obj1.std_avgre, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig3, ax3, obj1.noisestr[0], obj1.avg_avgdist, yerr=obj1.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="red", label_name=labels[1], legend=True)
        add_one_var_fig(fig4, ax4, obj1.noisestr[0], obj1.avg_snr, yerr=obj1.std_snr, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig1, ax1, obj2.noisestr[0], obj2.avg_avgcc, yerr=obj2.std_avgcc, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig2, ax2, obj2.noisestr[0], obj2.avg_avgre, yerr=obj2.std_avgre, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig3, ax3, obj2.noisestr[0], obj2.avg_avgdist, yerr=obj2.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="blue", label_name=labels[2], legend=True)
        add_one_var_fig(fig4, ax4, obj2.noisestr[0], obj2.avg_snr, yerr=obj2.std_snr, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig1, ax1, obj3.noisestr[0], obj3.avg_avgcc, yerr=obj3.std_avgcc, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj3.noisestr[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj3.noisestr[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="green", label_name=labels[3], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj3.noisestr[0], obj3.avg_snr, yerr=obj3.std_snr, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_snr.png")

    plt.show()

    return


def merge_two(input1, input2, exportname):
    obj1 = read_obj(input1)
    obj2 = read_obj(input2)

    for i, j in enumerate(obj1.names):
        for k in range(len(obj2.spontnoise[i])):
            obj1.avgcc[i].append(obj2.avgcc[i][k])
            obj1.avgdist[i].append(obj2.avgdist[i][k])
            obj1.avgre[i].append(obj2.avgre[i][k])
            obj1.avgsnr[i].append(obj2.avgsnr[i][k])
            obj1.avgsnrdb[i].append(obj2.avgsnrdb[i][k])
            obj1.cc[i].append(obj2.cc[i][k])
            obj1.dist[i].append(obj2.dist[i][k])
            obj1.noisestr[i].append(obj2.noisestr[i][k])
            obj1.re[i].append(obj2.re[i][k])
            obj1.snr[i].append(obj2.snr[i][k])
            obj1.snrdb[i].append(obj2.snrdb[i][k])
            obj1.spontnoise[i].append(obj2.spontnoise[i][k])

    obj1.save_obj(exportname)

    return


def simulate_rms(filename="default", system_glob="OPM", components=["rad", "tan", "ver"], magnetometer_type=1):

    import matplotlib.pyplot as plt
    # standard_path = '/home/marhl/Nextcloud/AEF_dataset/'
    data_path = 'data/'
    freesurfer_path = data_path + '/subjects/'

    block_name_opm = None
    block_name_squid = None

    # We import the data Measurements.txt, this part is for automated import of measurements
    line = 2
    measurements = np.loadtxt(data_path + "/Measurements.txt", delimiter=";", skiprows=1, dtype=str)
    start = line

    subjects = ["tisa", "urma", "daha", "kasc", "anjo", "peho", "miwl", "taya"]
    gen12s = [2, 2, 1, 1, 1, 1, 1, 1]

    pos = "both"
    number_of_noises = 100

    rmss = np.zeros((len(subjects), number_of_noises))

    for j, sub in enumerate(subjects):
        for i in range(start, len(measurements[:, 0])):
            meas_line = i
            system = measurements[meas_line, 3]
            gain = float(measurements[meas_line, 4])
            t_delay = float(measurements[meas_line, 5])
            block_name = measurements[meas_line, 1]
            status = measurements[meas_line, 0]
            comment = measurements[meas_line, 7]
            act_shield = measurements[meas_line, 2]

            if ((status == "2") and (sub in block_name)):
                print(system, gain, t_delay, block_name)
                if system == "SQUID":
                    block_name_squid = block_name
                    name = block_name[0:4]
                if system == "OPM":
                    block_name_opm = block_name
                    name = block_name[0:4]

        src_path = data_path + '/MNE/' + name + '-oct6-src.fif'
        src = mne.read_source_spaces(src_path)

        fname_bem = data_path + '/MNE/' + name + '-5120-5120-5120-bem-sol.fif'
        bem = mne.read_bem_solution(fname_bem)

        if system_glob == "SQUID":
            evoked_opm, dip_sim = simulate_aef_squid_mnepython(data_path, block_name_squid,
                                                               freesurfer_path,
                                                               position=pos,
                                                               magnetometer_type=magnetometer_type,
                                                               no_noises=number_of_noises, bem=bem, src=src)

        else:
            evoked_opm, dip_sim = simulate_aef_opm_mnepython(data_path, block_name_opm, freesurfer_path,
                                                             position=pos, gen12=gen12s[j],
                                                             no_noises=number_of_noises,
                                                             components=components, bem=bem, src=src)

        rmss[j,:] = np.sqrt(np.mean(np.square(evoked_opm.data), axis=0))

    print(rmss)
    ave_rmss = np.mean(rmss, axis=1)
    ave_rmss_subjects = np.mean(ave_rmss)
    ave_rmss2 = np.mean(rmss)
    ave_std2 = np.std(rmss)
    ave_std_subjects = np.std(rmss, axis=1)
    ave_std = np.std(ave_rmss)

    return 0


def simulate(filename="default", system_glob="OPM", components=["rad", "tan", "ver"], magnetometer_type=1, n_jobs=1):

    import matplotlib.pyplot as plt
    # standard_path = '/home/marhl/Nextcloud/AEF_dataset/'
    data_path = 'data/'
    freesurfer_path = data_path + '/subjects/'

    block_name_opm = None
    block_name_squid = None

    # We import the data Measurements.txt, this part is for automated import of measurements
    line = 2
    measurements = np.loadtxt(data_path + "/Measurements.txt", delimiter=";", skiprows=1, dtype=str)
    start = line

    subjects = ["tisa", "urma", "daha", "kasc", "anjo", "peho", "miwl", "taya"]
    gen12s = [2, 2, 1, 1, 1, 1, 1, 1]

    continue_session = False
    from os.path import exists
    if exists(filename):
        print('Should I overwrite previous session, else continue (Y/N):')
        x = input()
        if (x == "Y" or x == "y"):
            avg_stat = AvgStatistics()
        elif (x == "N" or x == "n"):
            avg_stat = read_obj(filename)
            continue_session = True
        else:
            print('Incorrect answer')
            return
    else:
        avg_stat = AvgStatistics()

    for j, sub in enumerate(subjects):
        for i in range(start, len(measurements[:, 0])):
            meas_line = i
            system = measurements[meas_line, 3]
            gain = float(measurements[meas_line, 4])
            t_delay = float(measurements[meas_line, 5])
            block_name = measurements[meas_line, 1]
            status = measurements[meas_line, 0]
            comment = measurements[meas_line, 7]
            act_shield = measurements[meas_line, 2]

            if ((status == "2") and (sub in block_name)):
                print(system, gain, t_delay, block_name)
                if system == "SQUID":
                    block_name_squid = block_name
                    name = block_name[0:4]
                if system == "OPM":
                    block_name_opm = block_name
                    name = block_name[0:4]

        if continue_session:
            if name not in avg_stat.names:
                avg_stat.add_name(name)
        else:
            avg_stat.add_name(name)

        pos = "both"
        number_of_noises = 100
        # number_of_noises = 4

        src_path = data_path + '/MNE/' + name + '-oct6-src.fif'
        src = mne.read_source_spaces(src_path)

        fname_bem = data_path + '/MNE/' + name + '-5120-5120-5120-bem-sol.fif'
        bem = mne.read_bem_solution(fname_bem)

        # brain = np.concatenate((src[0]["rr"], src[1]["rr"]))
        # center = [(max(brain[:, 0]) + min(brain[:, 0])) / 2., (max(brain[:, 1]) + min(brain[:, 1])) / 2.,
        # 		  (max(brain[:, 2]) + min(brain[:, 2])) / 2.]
        # bem = mne.make_sphere_model(r0=center)

        # noise_std_base = 75
        # for ii in np.arange(1.0, 1.5, 0.05):
        #     noise_std = noise_std_base * ii
        for ii in range(22, 0, -3):  # CAN ALSO BE SNR
            noise_std = float(ii)
        # for ii in range(0, 200, 25):
        # for ii in range(0, 25, 25):
        #    noise_std = float(ii) * 1.0 * 10 ** (-15)
            # for iii in range(0, 8, 1):
            for iii in range(0, 1, 1):
                spont_nois_dip = float(iii) * 1.0 * 10 ** (-9)
                print ("current: " + name + ", random noise: " + str(noise_std) + ", spon. noise: " + str(spont_nois_dip))

                if continue_session:
                    if noise_std in avg_stat.noisestr[j] and spont_nois_dip in avg_stat.spontnoise[j]:
                        break

                if system_glob == "SQUID":
                    evoked_opm, dip_sim = simulate_aef_squid_mnepython(data_path, block_name_squid,
                                                                       freesurfer_path,
                                                                       position=pos,
                                                                       magnetometer_type=magnetometer_type,
                                                                       no_noises=number_of_noises, bem=bem, src=src)

                else:
                    evoked_opm, dip_sim = simulate_aef_opm_mnepython(data_path, block_name_opm, freesurfer_path,
                                                                     position=pos, gen12=gen12s[j],
                                                                     no_noises=number_of_noises,
                                                                     components=components, bem=bem, src=src)

                # evoked_opm_noised, random_noises = add_noise_random(evoked_opm, noise_std)
                evoked_opm_noised, random_noises = add_noise_random_str(evoked_opm, noise_std)

                evoked_opm_noised2, spont_noises = add_noise_spontanous(evoked_opm_noised, spont_nois_dip,
                                                                        evoked_opm.times,
                                                                        data_path, name, bem=bem, src=src)

                # plot_evokedobj_topo_v3(evoked_opm, "", block_name_opm, "OPM", 0.0, freesurfer_path)
                # plot_evokedobj_topo_v3(evoked_opm_noised2, standard_path, block_name_opm, "OPM", 0.0, freesurfer_path)
                # plot_evokedobj_topo(evoked_opm_noised2, "", block_name_squid, "SQUID", 0.0)
                # plt.savefig("system4.png")
                # plt.show()

                fit_dip_opm, recon_field_opm = localize_dip(evoked_opm_noised2, data_path, name, bem=bem, n_jobs=n_jobs)

                dist, avg_dist = dipole_distance(dip_sim, fit_dip_opm)
                rel_err, corr_coeff, rel_err_avg, corr_coeff_avg = recc_two_evoked(evoked_opm_noised2, recon_field_opm)
                SNR_db_avgs, SNR_avgs, SNR_db, SNR = estimate_snr(evoked_opm, [random_noises, spont_noises])

                # avg_dist = random.random()
                # rel_err_avg = random.random()
                # corr_coeff_avg = random.random()
                # SNR_avgs = random.random()
                # SNR_db_avgs = random.random()
                #
                # dist = np.random.random_sample(number_of_noises)
                # rel_err = np.random.random_sample(number_of_noises)
                # corr_coeff = np.random.random_sample(number_of_noises)
                # SNR = np.random.random_sample(number_of_noises)
                # SNR_db = np.random.random_sample(number_of_noises)


                avg_stat.add_avgdist(avg_dist, j)
                avg_stat.add_avgre(rel_err_avg, j)
                avg_stat.add_avgcc(corr_coeff_avg, j)
                avg_stat.add_avgsnr(SNR_avgs, j)
                avg_stat.add_avgsnrdb(SNR_db_avgs, j)

                avg_stat.add_spontnoise(spont_nois_dip, j)
                avg_stat.add_noisestr(noise_std, j)

                avg_stat.add_dists(dist, j)
                avg_stat.add_res(rel_err, j)
                avg_stat.add_ccs(corr_coeff, j)
                avg_stat.add_snrs(SNR, j)
                avg_stat.add_snr_dbs(SNR_db, j)

                avg_stat.save_obj(filename)

    return 0


def test(fname, fname1=None, fname2=None, fname3=None, labels=["empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()

    return


def plot_contour(fname, fname1=None, fname2=None, fname3=None, labels=["empty", "empty", "empty", "empty"], parameter="dist"):
    import matplotlib.pyplot as plt
    import matplotlib.mlab as ml
    import numpy as np
    from scipy.interpolate import griddata

    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()
    z = obj.avg_avgdist

    z1 = np.zeros(np.shape(z))
    z2 = np.zeros(np.shape(z))
    z3 = np.zeros(np.shape(z))
    instances = 1

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.calc_avgs()
        obj1.calc_stds()
        z1 = obj1.avg_avgdist
        instances += 1

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.calc_avgs()
        obj2.calc_stds()
        z2 = obj2.avg_avgdist
        instances += 1

    if isinstance(fname2, str):
        obj3 = read_obj(fname3)
        obj3.calc_avgs()
        obj3.calc_stds()
        z3 = obj3.avg_avgdist
        instances += 1

    inf_zero=0
    if parameter=="dist":
        zlabel = "$d(p1, p2) \mathrm{[mm]}$"
        z = obj.avg_avgdist
        z1 = obj1.avg_avgdist
        z2 = obj2.avg_avgdist
        z3 = obj3.avg_avgdist
        zmax = np.max((z, z1, z2, z3))
        zmin = np.min((z, z1, z2, z3))
    elif parameter=="re":
        zlabel = '$\mathrm{RE}$'
        z = obj.avg_avgre
        z1 = obj1.avg_avgre
        z2 = obj2.avg_avgre
        z3 = obj3.avg_avgre
        zmax = np.max((z, z1, z2, z3))
        zmin = np.min((z, z1, z2, z3))
    elif parameter=="cc":
        zlabel = "$\mathrm{CC}$"
        z = obj.avg_avgcc
        z1 = obj1.avg_avgcc
        z2 = obj2.avg_avgcc
        z3 = obj3.avg_avgcc
        zmax = np.max((z, z1, z2, z3))
        zmin = np.min((z, z1, z2, z3))
    elif parameter=="snr":
        zlabel = "$SNR \mathrm{[dB]}$"
        z = obj.avg_snrdb
        z1 = obj1.avg_snrdb
        z2 = obj2.avg_snrdb
        z3 = obj3.avg_snrdb
        for ii, jj in enumerate(z):
            if not np.isfinite(z[ii]):
                z[ii] = 100
                z1[ii] = 100
                z2[ii] = 100
                z3[ii] = 100
                inf_zero = 1

    if inf_zero == 1:
        zmax = np.max((z[1:], z1[1:], z2[1:], z3[1:]))
        zmin = np.min((z[1:], z1[1:], z2[1:], z3[1:]))
    else:
        zmax = np.max((z, z1, z2, z3))
        zmin = np.min((z, z1, z2, z3))

    nx = 100
    ny = 100
    x = np.average(np.array(obj.noisestr)*10**(15), axis=0)
    # un_x = np.unique(x)
    # x = np.reshape(x, (len(un_x),-1))
    xmin = np.min(x)
    xmax = np.max(x)
    y = np.average(np.array(obj.spontnoise)*10**(9), axis=0)
    # un_y = np.unique(y)
    # y = np.reshape(y, (len(un_y),-1))
    ymin = np.min(y)
    ymax = np.max(y)

    for i_inst in range(instances):
        if i_inst==0:
            z = z
        if i_inst==1:
            z = z1
        if i_inst==2:
            z = z2
        if i_inst==3:
            z = z3

        xi = np.linspace(xmin, xmax, nx)
        yi = np.linspace(ymin, ymax, ny)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi, yi), method='cubic')

        plt.contour(xi, yi, zi, levels=np.linspace(zmin,zmax,15), linewidths=0.5, colors='k')
        plt.pcolormesh(xi, yi, zi, cmap=plt.get_cmap('rainbow'), vmin=zmin, vmax=zmax)

        clb = plt.colorbar()
        plt.scatter(x, y, marker='o', c='b', s=10, zorder=10)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("$\sigma \mathrm{[fT]}$")
        plt.ylabel("$\| q_{\mathrm{spont}} \|  \mathrm{[nAm]}$")
        clb.ax.set_title(zlabel)
        plt.savefig(fname+labels[i_inst]+"-"+parameter+".png")
        plt.show()

    return


def visualize(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt

    if isinstance(fname, str):
        obj = read_obj(fname)
        obj.calc_avgs()
        obj.calc_avgs2()
        obj.calc_stds()
        obj.calc_stds2()

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.calc_avgs()
        obj1.calc_stds()
        obj1.calc_stds2()


    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.calc_avgs()
        obj2.calc_stds()
        obj2.calc_stds2()

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.calc_avgs()
        obj3.calc_stds()
        obj3.calc_stds2()

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.calc_avgs()
        obj4.calc_stds()
        obj4.calc_stds2()

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.calc_avgs()
        obj5.calc_stds()
        obj5.calc_stds2()

    fig1, ax1 = plot_one_var(obj.noisestr[0], obj.avg_avgcc, yerr=obj.std_avgcc, xname="$\sigma \mathrm{[fT]}$",
                             yname="$\mathrm{CC}$", xscale=10 ** 15, label_name=labels[0] , legend=True)
    fig2, ax2 = plot_one_var(obj.noisestr[0], obj.avg_avgre, yerr=obj.std_avgre, xname="$\sigma \mathrm{[fT]}$",
                             yname="$\mathrm{RE}$", xscale=10 ** 15, label_name=labels[0] , legend=True)
    fig3, ax3 = plot_one_var(obj.noisestr[0], obj.avg_avgdist, yerr=obj.std_avgdist, xname="$\sigma \mathrm{[fT]}$",
                             yname="$d(p1, p2) \mathrm{[mm]}$", xscale=10 ** 15, yscale=10 ** 3,
                             label_name=labels[0] , legend=True)
    fig4, ax4 = plot_one_var(obj.noisestr[0], obj.avg_snr, yerr=obj.std_snr, xname="$\sigma \mathrm{[fT]}$",
                             yname="$\mathrm{SNR}$", xscale=10 ** 15, label_name=labels[0] , legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig1, ax1, obj1.noisestr[0], obj1.avg_avgcc, yerr=obj1.std_avgcc, xscale=10 ** 15, color="red",
                        label_name=labels[1] , legend=True)
        add_one_var_fig(fig2, ax2, obj1.noisestr[0], obj1.avg_avgre, yerr=obj1.std_avgre, xscale=10 ** 15, color="red",
                        label_name=labels[1] , legend=True)
        add_one_var_fig(fig3, ax3, obj1.noisestr[0], obj1.avg_avgdist, yerr=obj1.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="red", label_name=labels[1] , legend=True)
        add_one_var_fig(fig4, ax4, obj1.noisestr[0], obj1.avg_snr, yerr=obj1.std_snr, xscale=10 ** 15, color="red",
                        label_name=labels[1] , legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig1, ax1, obj2.noisestr[0], obj2.avg_avgcc, yerr=obj2.std_avgcc, xscale=10 ** 15, color="blue",
                        label_name=labels[2] , legend=True)
        add_one_var_fig(fig2, ax2, obj2.noisestr[0], obj2.avg_avgre, yerr=obj2.std_avgre, xscale=10 ** 15, color="blue",
                        label_name=labels[2] , legend=True)
        add_one_var_fig(fig3, ax3, obj2.noisestr[0], obj2.avg_avgdist, yerr=obj2.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="blue", label_name=labels[2] , legend=True)
        add_one_var_fig(fig4, ax4, obj2.noisestr[0], obj2.avg_snr, yerr=obj2.std_snr, xscale=10 ** 15, color="blue",
                        label_name=labels[2] , legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig1, ax1, obj3.noisestr[0], obj3.avg_avgcc, yerr=obj3.std_avgcc, xscale=10 ** 15, color="green",
                        label_name=labels[3] , legend=True, savefig=labels[0] + "_components_cc_std1.png")
        add_one_var_fig(fig2, ax2, obj3.noisestr[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=10 ** 15, color="green",
                        label_name=labels[3] , legend=True, savefig=labels[0] + "_components_re_std1.png")
        add_one_var_fig(fig3, ax3, obj3.noisestr[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="green", label_name=labels[3] , legend=True,
                        savefig=labels[0] + "_components_dist_std1.png")
        add_one_var_fig(fig4, ax4, obj3.noisestr[0], obj3.avg_snr, yerr=obj3.std_snr, xscale=10 ** 15, color="green",
                        label_name=labels[3] , legend=True, savefig=labels[0] + "_components_snr_std1.png")

    if isinstance(fname4, str):
        add_one_var_fig(fig1, ax1, obj4.noisestr[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=10 ** 15, color="cyan",
                        label_name=labels[4] , legend=True)
        add_one_var_fig(fig2, ax2, obj4.noisestr[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=10 ** 15, color="cyan",
                        label_name=labels[4] , legend=True)
        add_one_var_fig(fig3, ax3, obj4.noisestr[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="cyan", label_name=labels[4] , legend=True)
        add_one_var_fig(fig4, ax4, obj4.noisestr[0], obj4.avg_snr, yerr=obj4.std_snr, xscale=10 ** 15, color="cyan",
                        label_name=labels[4] , legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig1, ax1, obj5.noisestr[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=10 ** 15, color="pink",
                        label_name=labels[5] , legend=True, savefig=labels[0] + "_components_cc_std1.png")
        add_one_var_fig(fig2, ax2, obj5.noisestr[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=10 ** 15, color="pink",
                        label_name=labels[5] , legend=True, savefig=labels[0] + "_components_re_std1.png")
        add_one_var_fig(fig3, ax3, obj5.noisestr[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="pink", label_name=labels[5] , legend=True,
                        savefig=labels[0] + "_components_dist_std1.png")
        add_one_var_fig(fig4, ax4, obj5.noisestr[0], obj5.avg_snr, yerr=obj5.std_snr, xscale=10 ** 15, color="pink",
                        label_name=labels[5] , legend=True, savefig=labels[0] + "_components_snr_std1.png")

    plt.show()

    return


def visualize2(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.calc_avgs()
        obj1.calc_stds()

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.calc_avgs()
        obj2.calc_stds()

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.calc_avgs()
        obj3.calc_stds()

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.calc_avgs()
        obj4.calc_stds()

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.calc_avgs()
        obj5.calc_stds()

    # xscale = 10 ** 15
    xscale = 10 ** 9
    # xname = "$\sigma \mathrm{[fT]}$"
    xname = "$\| q_{\mathrm{spont}} \|  \mathrm{[nAm]}$"

    fig1, ax1 = plot_one_var(obj.spontnoise[0], obj.avg_avgcc, yerr=obj.std_avgcc, xname=xname, yname="$\mathrm{CC}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig2, ax2 = plot_one_var(obj.spontnoise[0], obj.avg_avgre, yerr=obj.std_avgre, xname=xname, yname="$\mathrm{RE}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig3, ax3 = plot_one_var(obj.spontnoise[0], obj.avg_avgdist, yerr=obj.std_avgdist, xname=xname,
                             yname="$d(p1, p2) \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=True)
    fig4, ax4 = plot_one_var(obj.spontnoise[0], obj.avg_snr, yerr=obj.std_snr, xname=xname, yname="$\mathrm{SNR}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig1, ax1, obj1.spontnoise[0], obj1.avg_avgcc, yerr=obj1.std_avgcc, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig2, ax2, obj1.spontnoise[0], obj1.avg_avgre, yerr=obj1.std_avgre, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig3, ax3, obj1.spontnoise[0], obj1.avg_avgdist, yerr=obj1.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="red", label_name=labels[1], legend=True)
        add_one_var_fig(fig4, ax4, obj1.spontnoise[0], obj1.avg_snr, yerr=obj1.std_snr, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig1, ax1, obj2.spontnoise[0], obj2.avg_avgcc, yerr=obj2.std_avgcc, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig2, ax2, obj2.spontnoise[0], obj2.avg_avgre, yerr=obj2.std_avgre, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig3, ax3, obj2.spontnoise[0], obj2.avg_avgdist, yerr=obj2.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="blue", label_name=labels[2], legend=True)
        add_one_var_fig(fig4, ax4, obj2.spontnoise[0], obj2.avg_snr, yerr=obj2.std_snr, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig1, ax1, obj3.spontnoise[0], obj3.avg_avgcc, yerr=obj3.std_avgcc, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj3.spontnoise[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj3.spontnoise[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="green", label_name=labels[3], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj3.spontnoise[0], obj3.avg_snr, yerr=obj3.std_snr, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_snr.png")

    if isinstance(fname4, str):
        add_one_var_fig(fig1, ax1, obj4.spontnoise[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj4.spontnoise[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj4.spontnoise[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="cyan", label_name=labels[4], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj4.spontnoise[0], obj4.avg_snr, yerr=obj4.std_snr, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_snr.png")

    if isinstance(fname5, str):
        add_one_var_fig(fig1, ax1, obj5.spontnoise[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj5.spontnoise[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj5.spontnoise[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="pink", label_name=labels[5], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj5.spontnoise[0], obj5.avg_snr, yerr=obj5.std_snr, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_snr.png")

    plt.show()

    return


def visualize3(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.calc_avgs()
        obj1.calc_stds()

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.calc_avgs()
        obj2.calc_stds()

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.calc_avgs()
        obj3.calc_stds()

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.calc_avgs()
        obj4.calc_stds()

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.calc_avgs()
        obj5.calc_stds()

    # xscale = 10 ** 15
    xscale = 1
    # xname = "$\sigma \mathrm{[fT]}$"
    xname = "$ SNR   \mathrm{[dB]}$"

    fig1, ax1 = plot_one_var(obj.noisestr[0], obj.avg_avgcc, yerr=obj.std_avgcc, xname=xname, yname="$\mathrm{CC}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig2, ax2 = plot_one_var(obj.noisestr[0], obj.avg_avgre, yerr=obj.std_avgre, xname=xname, yname="$\mathrm{RE}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig3, ax3 = plot_one_var(obj.noisestr[0], obj.avg_avgdist, yerr=obj.std_avgdist, xname=xname,
                             yname="$d(p1, p2) \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=True)
    fig4, ax4 = plot_one_var(obj.noisestr[0], obj.avg_snr, yerr=obj.std_snr, xname=xname, yname="$\mathrm{SNR}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig1, ax1, obj1.noisestr[0], obj1.avg_avgcc, yerr=obj1.std_avgcc, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig2, ax2, obj1.noisestr[0], obj1.avg_avgre, yerr=obj1.std_avgre, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig3, ax3, obj1.noisestr[0], obj1.avg_avgdist, yerr=obj1.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="red", label_name=labels[1], legend=True)
        add_one_var_fig(fig4, ax4, obj1.noisestr[0], obj1.avg_snr, yerr=obj1.std_snr, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig1, ax1, obj2.noisestr[0], obj2.avg_avgcc, yerr=obj2.std_avgcc, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig2, ax2, obj2.noisestr[0], obj2.avg_avgre, yerr=obj2.std_avgre, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig3, ax3, obj2.noisestr[0], obj2.avg_avgdist, yerr=obj2.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="blue", label_name=labels[2], legend=True)
        add_one_var_fig(fig4, ax4, obj2.noisestr[0], obj2.avg_snr, yerr=obj2.std_snr, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig1, ax1, obj3.noisestr[0], obj3.avg_avgcc, yerr=obj3.std_avgcc, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj3.noisestr[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj3.noisestr[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="green", label_name=labels[3], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj3.noisestr[0], obj3.avg_snr, yerr=obj3.std_snr, xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_snr.png")

    if isinstance(fname4, str):
        add_one_var_fig(fig1, ax1, obj4.noisestr[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj4.noisestr[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj4.noisestr[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="cyan", label_name=labels[4], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj4.noisestr[0], obj4.avg_snr, yerr=obj4.std_snr, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_snr.png")

    if isinstance(fname5, str):
        add_one_var_fig(fig1, ax1, obj5.noisestr[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj5.noisestr[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj5.noisestr[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="pink", label_name=labels[5], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj5.noisestr[0], obj5.avg_snr, yerr=obj5.std_snr, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_snr.png")

    ax1.invert_xaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()
    ax4.invert_xaxis()

    plt.show()

    return


def visualize4(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.calc_avgs()
        obj1.calc_stds()

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.calc_avgs()
        obj2.calc_stds()

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.calc_avgs()
        obj3.calc_stds()

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.calc_avgs()
        obj4.calc_stds()

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.calc_avgs()
        obj5.calc_stds()

    xscale = 10 ** 15
    # xscale = 1
    xname = "$\sigma \mathrm{[fT]}$"
    # xname = "$\| SNR \|  \mathrm{[dB]}$"

    # np.argwhere(np.array(obj.spontnoise[0]) == 0.0).tolist()
    res = [idx for idx, val in enumerate(obj.spontnoise[0]) if val == 0.0]

    fig1, ax1 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_avgcc)[res], yerr=np.array(obj.std_avgcc)[res], xname=xname, yname="$\mathrm{CC}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig2, ax2 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_avgre)[res], yerr=np.array(obj.std_avgre)[res], xname=xname, yname="$\mathrm{RE}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig3, ax3 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_avgdist)[res], yerr=np.array(obj.std_avgdist)[res], xname=xname,
                             yname="$d(p1, p2) \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=True)
    fig4, ax4 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_snr)[res], yerr=np.array(obj.std_snr)[res], xname=xname, yname="$\mathrm{SNR}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig1, ax1, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_avgcc)[res], yerr=np.array(obj1.std_avgcc)[res], xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig2, ax2, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_avgre)[res], yerr=np.array(obj1.std_avgre)[res], xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig3, ax3, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_avgdist)[res], yerr=np.array(obj1.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color="red", label_name=labels[1], legend=True)
        add_one_var_fig(fig4, ax4, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_snr)[res], yerr=np.array(obj1.std_snr)[res], xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig1, ax1, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_avgcc)[res], yerr=np.array(obj2.std_avgcc)[res], xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig2, ax2, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_avgre)[res], yerr=np.array(obj2.std_avgre)[res], xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig3, ax3, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_avgdist)[res], yerr=np.array(obj2.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color="blue", label_name=labels[2], legend=True)
        add_one_var_fig(fig4, ax4, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_snr)[res], yerr=np.array(obj2.std_snr)[res], xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig1, ax1, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_avgcc)[res], yerr=np.array(obj3.std_avgcc)[res], xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_avgre)[res], yerr=np.array(obj3.std_avgre)[res], xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_avgdist)[res], yerr=np.array(obj3.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3,
                        color="green", label_name=labels[3], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_snr)[res], yerr=np.array(obj3.std_snr)[res], xscale=xscale, color="green",
                        label_name=labels[3], legend=True, savefig=labels[0] + "_components_snr.png")

    if isinstance(fname4, str):
        add_one_var_fig(fig1, ax1, obj4.noisestr[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj4.noisestr[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj4.noisestr[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="cyan", label_name=labels[4], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj4.noisestr[0], obj4.avg_snr, yerr=obj4.std_snr, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True, savefig=labels[0] + "_components_snr.png")

    if isinstance(fname5, str):
        add_one_var_fig(fig1, ax1, obj5.noisestr[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_cc.png")
        add_one_var_fig(fig2, ax2, obj5.noisestr[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_re.png")
        add_one_var_fig(fig3, ax3, obj5.noisestr[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="pink", label_name=labels[5], legend=True,
                        savefig=labels[0] + "_components_dist.png")
        add_one_var_fig(fig4, ax4, obj5.noisestr[0], obj5.avg_snr, yerr=obj5.std_snr, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True, savefig=labels[0] + "_components_snr.png")

    plt.show()

    return


def estimate_snr(evoked_sig, evoked_noise, noise_times=[], signal_times=[]):
    evoked1 = evoked_sig.copy()

    if isinstance(evoked_noise, list):
        evoked2 = evoked_noise[0].copy()
        for i in range(1, len(evoked_noise)):
            evoked2.data = evoked2.data + evoked_noise[i].data
    else:
        evoked2 = evoked_noise.copy()

    if len(signal_times) > 0:
        evoked1.crop(min(signal_times), max(signal_times))

    if len(noise_times) > 0:
        evoked2.crop(min(noise_times), max(noise_times))

    rms_signal = np.sqrt(np.mean(np.square(evoked1.data), axis=0))
    rms_noise = np.sqrt(np.mean(np.square(evoked2.data), axis=0))

    # rms_signal_nonavg = np.sqrt(np.square(evoked1.data))
    # rms_noise_nonavg = np.sqrt(np.square(evoked2.data))

    SNR_db = 20 * np.log10(np.mean(rms_signal) / np.mean(rms_noise))
    SNR = np.mean(rms_signal) / np.mean(rms_noise)

    # SNR_db_nonavg = np.mean(20 * np.log10(rms_signal_nonavg / rms_noise_nonavg), axis=0)
    # SNR_nonavg = np.mean(rms_signal_nonavg / rms_noise_nonavg, axis=0)

    return SNR_db, SNR, rms_signal, rms_noise #SNR_db_nonavg, SNR_nonavg


def recc_two_evoked(evoked_1, evoked_2):
    import megtools.vector_functions as vfun

    evoked1 = evoked_1.copy()
    evoked2 = evoked_2.copy()

    mag_evoked1 = evoked1.data
    mag_evoked2 = evoked2.data

    rel_err = []
    corr_coeff = []
    for i in range(len(evoked1.times)):
        rel_err.append(vfun.rel_err_vojko(mag_evoked1[:, i], mag_evoked2[:, i]))
        corr_coeff.append(vfun.corr_coeff_vojko(mag_evoked1[:, i], mag_evoked2[:, i]))

    rel_err = np.array(rel_err)
    corr_coeff = np.array(corr_coeff)
    rel_err_avg = np.mean(rel_err)
    corr_coeff_avg = np.mean(corr_coeff)

    return rel_err, corr_coeff, rel_err_avg, corr_coeff_avg


def dipole_distance(dip1, dip2):
    dist = np.sqrt((dip1.pos[:, 0] - dip2.pos[:, 0]) ** 2 + (dip1.pos[:, 1] - dip2.pos[:, 1]) ** 2 + (
                dip1.pos[:, 2] - dip2.pos[:, 2]) ** 2)
    avg_dist = np.mean(dist)

    return dist, avg_dist


def add_noise_random(evoked, noise_std):
    import numpy
    import copy

    evoked_noised = evoked.copy()
    evoked_noises = evoked.copy()
    random_normal = numpy.random.normal(loc=0.0, scale=noise_std, size=np.shape(evoked_noised.data))
    evoked_noised.data = evoked_noised.data + random_normal
    evoked_noises.data = random_normal

    return evoked_noised, evoked_noises


def add_noise_random_str(evoked, snr):
    import numpy
    import copy

    evoked_noised = evoked.copy()
    evoked_noises = evoked.copy()

    rms_opm_signal = np.mean(np.sqrt(np.mean(np.square(evoked_noised.data), axis=0)))
    rms_opm_noise = rms_opm_signal / (10 ** (snr / 20))

    noise = np.random.normal(0, rms_opm_noise, evoked_noised.data.shape)

    evoked_noised.data = evoked_noised.data + noise
    evoked_noises.data = noise

    return evoked_noised, evoked_noises


def add_noise_spontanous(evoked, dip_str, times, standard_path, name, no_dip=100, bem=None, src=None):
    import mne

    coil_def_fname = 'data/coil_def_custom.dat'
    subject_dir = standard_path + '/FreeSurferProject/subjects/'
    if bem == None:
        bem = standard_path + '/MNE/' + name + '/' + name + '-5120-5120-5120-bem-sol.fif'
    if src == None:
        src_path = standard_path + '/MNE/' + name + '/' + name + '-oct6-src.fif'
        src = mne.read_source_spaces(src_path)

    evoked_noised = evoked.copy()
    for i, j in enumerate(times):
        if i == 0:
            dip_times = np.array(no_dip * [j])
        else:
            dip_times = np.hstack((dip_times, np.array(no_dip * [j])))

    dip_loc, dip_str = generate_dip(src, dip_times, dip_str)
    dipoles = mne.Dipole(dip_times, dip_loc[:, 0:3], dip_str, dip_loc[:, 3:6], 1)

    # plot_dipoles(dip_loc[np.where(dip_times==0.0)[0].tolist()], subject_dir, name, savefig="spontanous_dipoles.png")

    with mne.use_coil_def(coil_def_fname):
        fwd, stc = mne.make_forward_dipole(dipoles, bem, evoked.info)
    noises = mne.simulation.simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf)

    evoked_noised.data = evoked_noised.data + noises.data

    return evoked_noised, noises


def generate_dip(src, times, max_dip, rand_dir=0, rand_dip=0):
    import random

    # np.random.seed(0)
    # random.seed(0)

    dip = np.zeros((len(times), 6))
    dir = np.zeros((len(times), 3))
    dip_str = np.ones((len(times))) * max_dip

    for i, j in enumerate(times):
        dir[i] = np.random.random_sample((3,)) * 2 - 1
        dir[i] = dir[i] / np.linalg.norm(dir[i])
        # dip[i] = np.hstack((src[1]["rr"][stc.rh_vertno][0], dir[i]))
        rand_hemi = random.randint(0, 1)
        rand = random.randint(0, len(src[rand_hemi]["rr"]) - 1)
        dip[i] = np.hstack((src[rand_hemi]["rr"][rand], dir[i]))

    # dir = np.random.random_sample((3,))
    # dip1 = np.hstack((src[0]["rr"][stc.lh_vertno][0], dir1))

    # rand = random.randint(0, len(src[0]["rr"])-1)
    # dip1 = np.hstack((src[0]["rr"][rand], dir1))
    # dip1_norad = vfun.rm_radial_component(dip1)
    # dip1_norad = np.copy(dip1)
    # dip1_norad[3:6] = dip1_norad[3:6] / np.linalg.norm(dip1_norad[3:6])

    return dip, dip_str


def localize_dip(evoked, standard_path, name, bem=None, n_jobs=1):
    import mne
    import matplotlib.pyplot as plt
    noise_covariance_opm_path = mne.make_ad_hoc_cov(evoked.info)
    snr = 1.
    noise_covariance_opm_path['data'] *= (1. / snr) ** 2
    trans = None

    if bem == None:
        bem = standard_path + '/MNE/' + name + '-5120-5120-5120-bem-sol.fif'

    coil_def_fname = 'data/coil_def_custom.dat'
    with mne.use_coil_def(coil_def_fname):
        dip, field_resid = mne.fit_dipole(evoked.copy(), noise_covariance_opm_path, bem, trans, n_jobs=n_jobs)

    reconst_field = evoked.copy()
    reconst_field.data = reconst_field.data - field_resid.data

    # dip.plot_locations(trans, name, subjects_dir, mode='orthoview')
    # dip_true.plot_locations(trans, name, subjects_dir, mode='orthoview')

    # plt.show()

    return dip, reconst_field


def simulate_aef_squid_mnepython(standard_path, block_name_squid, subject_dir, position="both",
                                 magnetometer_type=1, noisy_ecds=0, noise_ratio=0.1, no_noises=1, src=None, bem=None):
    # magnetometer_type: 1(axial gradiometer), 2(radial magnetometer), 3(planar gradiometer #1),
    # 4(planar gradiometer #2), 5(tangential magnetometer #1), 6(tangential magnetometer #2)
    import mne

    name = block_name_squid[0:4]

    if src is None:
        src_path = standard_path + '/MNE/' + name + '-oct6-src.fif'
        src = mne.read_source_spaces(src_path)

    if bem is None:
        bem = standard_path + '/MNE/' + name + '-5120-5120-5120-bem-sol.fif'

    fwd_squid_path = standard_path + "MNE/" + block_name_squid + '-fwd.fif'
    template_name = standard_path + "/ET160_template.0100.flt.hdr"
    fname_trans_squid = standard_path + '/MNE/' + block_name_squid + '-trans.fif'

    tang, planar, grad_plan, dir_plan = 0, 0, "xy", "xy"
    mag_num = 6001
    name_of_syst = "axial_gradiometer"

    if magnetometer_type == 1:
        tang, planar, grad_plan, dir_plan = 0, 0, "xy", "xy"
        mag_num = 6001
        name_of_syst = "axial_gradiometer"

    if magnetometer_type == 2:
        tang, planar, grad_plan, dir_plan = 0, 0, "xy", "xy"
        mag_num = 6002
        name_of_syst = "radial_magnetometer"

    if magnetometer_type == 3:
        tang, planar, grad_plan, dir_plan = 0, 1, "xy", "xy"
        mag_num = 6004
        name_of_syst = "planar_gradiometer_latitude"

    if magnetometer_type == 4:
        tang, planar, grad_plan, dir_plan = 0, 1, "yz", "xy"
        mag_num = 6004
        name_of_syst = "planar_gradiometer_longitude"

    if magnetometer_type == 5:
        tang, planar, grad_plan, dir_plan = 1, 0, "xy", "xy"
        mag_num = 6002
        name_of_syst = "tangential_magnetometer_latitude"

    if magnetometer_type == 6:
        tang, planar, grad_plan, dir_plan = 1, 0, "yz", "yz"
        mag_num = 6002
        name_of_syst = "tangential_magnetometer_longitude"

    xyz1, xyz2, rot_matrix = import_sensor_pos_squid2(template_name, fwd_squid_path, fname_trans_squid, name,
                                                      tangential=tang, planar_gradiometer=planar, grad_plane=grad_plan,
                                                      dir_plane=dir_plan)

    plot_magnetometers3(subject_dir, name, xyz1, rot_matrix, magnetometer_number=mag_num,
                        coil_def='data/coil_def_custom.dat', filename=name_of_syst)

    info = create_squids2(xyz1[:, 0:3], rot_matrix, mag_number=mag_num)

    # plot_magnetometers2(subject_dir, name, xyz1, rot_matrix, magnetometer_type=2)
    # plot_magnetometers(subject_dir, name, xyz1, xyz2, magnetometer_type=2)
    # plot_magnetometers2(subject_dir, name, xyz1, rot_matrix, magnetometer_type=2)

    times = np.arange(no_noises, dtype=float) * 0.02

    max_dip = 100.0 * 10 ** (-9)

    dip_loc, dip_str = generate_dip(src, times, max_dip)

    # plot_dipoles(xyz,dip_loc,subject_dir, name)

    dip = mne.Dipole(np.array(times), dip_loc[:, 0:3], dip_str, dip_loc[:, 3:6], 1)

    # plot_dipoles(dip_loc[np.where(times == 0.0)[0].tolist()], subject_dir, name, savefig="simulated_dip.png")

    coil_def_fname = 'data/coil_def_custom.dat'
    with mne.use_coil_def(coil_def_fname):
        fwd_dip, stc_dip = mne.make_forward_dipole(dip, bem, info, trans=None)
    evoked = mne.simulation.simulate_evoked(fwd_dip, stc_dip, info, cov=None, nave=np.inf)

    # coil_def_fname = 'data/coil_def_custom.dat'
    # with mne.use_coil_def(coil_def_fname):
    #     fig = mne.viz.plot_alignment(evoked.info, trans=None, subject=block_name_squid[0:4],
    #                                  subjects_dir=subject_dir, surfaces='head-dense',
    #                                  show_axes=True, dig=True, eeg=[], meg='sensors',
    #                                  coord_frame='meg', mri_fiducials='estimated')
    #     import mayavi.mlab
    #     mayavi.mlab.show()

    return evoked, dip


def simulate_aef_opm_mnepython(standard_path, block_name_opm, subject_dir, position="both", gen12=1,
                               no_noises=1, components=["rad", "tan", "ver"], bem=None, src=None):
    import mne
    import matplotlib.pyplot as plt
    import megtools.meg_plot as mplo
    import megtools.my3Dplot as m3p

    name = block_name_opm[0:4]
    sensorholder_path = "sensorholders/"

    # labels = mne.read_labels_from_annot(block_name_opm[0:4], subjects_dir=subject_dir, parc="aparc.a2009s", hemi="both")
    # label_names = ['G_temp_sup-G_T_transv-lh', 'G_temp_sup-G_T_transv-rh']
    # label_names = ['G_temp_sup-G_T_transv-rh']
    # aud_labels = [label for label in labels if label.name in label_names]

    if src is None:
        src_path = standard_path + '/MNE/' + name + '-oct6-src.fif'
        src = mne.read_source_spaces(src_path)

    if bem is None:
        bem = standard_path + '/MNE/' + name + '-5120-5120-5120-bem-sol.fif'

    xyz1 = import_sensor_pos_ori_all(sensorholder_path, name, subject_dir, gen12=gen12)
    xyz2 = []
    for i in range(0, len(xyz1), 2):
        xyz2.append(xyz1[i])
        xyz2.append(xyz1[i + 1])
        xyz = xyz1[i][0:3]
        dir = np.cross(xyz1[i][3:6], xyz1[i + 1][3:6])
        xyz2.append(np.hstack((xyz, dir)))
    xyz2 = np.array(xyz2).astype(float)
    xyz = xyz2.copy()
    xyz[0::3, 3:6] = -xyz[0::3, 3:6]
    xyz[1::3, 3:6] = -xyz[1::3, 3:6]

    info = create_opms(xyz, components=components)

    xyz_info = []
    rot_mat_info = []
    for j, i in enumerate(info.ch_names):
        xyz_info.append(info['chs'][j]['loc'][0:3])
        rot_mat_info.append(info['chs'][j]['loc'][3:12].reshape((3, 3)))
    xyz_info = np.array(xyz_info)
    rot_mat_info = np.array(rot_mat_info)

    # plot_magnetometers3(subject_dir, name, xyz_info, rot_mat_info, magnetometer_number=9999,
    #                     coil_def='data/coil_def_custom.dat')
    # plot_magnetometers2(subject_dir, name, xyz_info, rot_mat_info, "opm")

    times = np.arange(no_noises, dtype=float) * 0.02

    max_dip = 100.0 * 10 ** (-9)

    dip_loc, dip_str = generate_dip(src, times, max_dip)

    # plot_dipoles(xyz,dip_loc,subject_dir, name)

    dip = mne.Dipole(np.array(times), dip_loc[:, 0:3], dip_str, dip_loc[:, 3:6], 1)

    # plot_dipoles(dip_loc[np.where(times == 0.0)[0].tolist()], subject_dir, name, savefig="simulated_dip.png")

    coil_def_fname = 'data/coil_def.dat'
    with mne.use_coil_def(coil_def_fname):
        fwd_dip, stc_dip = mne.make_forward_dipole(dip, bem, info, trans=None)
    evoked = mne.simulation.simulate_evoked(fwd_dip, stc_dip, info, cov=None, nave=np.inf)

    return evoked, dip


def plot_dipoles(dip, subject_dir, name, savefig=None):
    import os
    import megtools.my3Dplot as m3p
    surface1 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_inner_skull_surface')
    surface2 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skull_surface')
    surface3 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skin_surface')

    surfaces = [surface1, surface2, surface3]
    # surfaces = [surface1]
    # surfaces = [surface1, surface3]

    # locations = xyz[:, 0:3]
    # unit_v = np.array([0., 0., 1.])
    # directions = np.zeros(np.shape(locations))
    # for i in range(len(locations)):
    # 	directions[i] = np.dot(unit_v, rot_mat[i])
    #
    # sensors = np.hstack((locations, directions))

    p1 = m3p.plot_sensors_pyvista(surfaces, sensors=dip, arrow_color="red")

    # p1.show(screenshot=name + 'opm_rad.png')
    # p1 = m3p.plot_sensors_pyvista(surfaces, sensors=[])
    if isinstance(savefig, str):
        p1.show(screenshot=savefig)
    else:
        p1.show()
    # p1.show()

    return


def plot_evokedobj_topo_v3(evoked, data_path, block_name, system, time, subject_dir, halve=False):
    import mne
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook
    import numpy as np
    import megtools.pyread_biosig as pbio
    import megtools.pymeg_visualize as pvis
    import megtools.vector_functions as vfun
    from scipy.interpolate import griddata
    import matplotlib.mlab as ml
    from mpl_toolkits.mplot3d import Axes3D
    from numpy import linalg as LA
    import megtools.meg_plot as mplo

    from matplotlib import rc
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    # import os
    # os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2015/bin/x86_64-darwin'
    # print(os.getenv("PATH"))

    if system == "OPM":
        # xyz_all = evoked.xyz
        channelnames = evoked.ch_names
        xyz_all = np.zeros((len(evoked.ch_names), 3))
        for i, j in enumerate(evoked.info['chs']):
            xyz_all[i] = j['loc'][0:3]

        # xx, yy = mplo.map_et_coord(-xyz_all[:, 0], xyz_all[:, 2], xyz_all[:, 1])
        # xx = (np.array(xx).reshape((-1, 1)))
        # yy = (np.array(yy).reshape((-1, 1)))
        # xy = np.concatenate((xx, yy), axis=1)

        xy = pvis.squid_xy_reconstruciton(xyz_all[:, 0:3])
        #		xy_orig = xy

        center = (np.mean(xy[:, 0]), np.mean(xy[:, 1]))
        radius = 1.2 * (np.max(xy[:, 0]) - np.min(xy[:, 0])) / 2.0

        picks = []
        orients = []  # tan or rad
        signs = []  # -1 or 1, if -1 it is the new generation

        # write prittier code
        for i, j in enumerate(channelnames):
            picks.append(i)
            if "rad" in j:
                orients.append("rad")
                signs.append(1)
            if "tan" in j:
                orients.append("tan")
                signs.append(1)
            if "ver" in j:
                orients.append("ver")
                signs.append(1)
            if "abs" in j:
                orients.append("abs")
                signs.append(1)

        rad_picks = np.where(np.array(orients) == "rad")[0].tolist()
        tan_picks = np.where(np.array(orients) == "tan")[0].tolist()
        ver_picks = np.where(np.array(orients) == "ver")[0].tolist()
        abs_picks = np.where(np.array(orients) == "abs")[0].tolist()

        xy_rad = xy[rad_picks]
        xy_tan = xy[tan_picks]
        xy_ver = xy[ver_picks]
        xy_abs = xy[abs_picks]

        chosen_time = np.argmin(np.abs(np.array(evoked.times) - time))

        mag = evoked.data[:, chosen_time]
        mag = mag * (10.0 ** 15.0)
        # print(signs)
        mag_rad = mag[rad_picks]
        mag_tan = mag[tan_picks]
        mag_ver = mag[ver_picks]
        mag_abs = mag[abs_picks]

        plot_both = 0
        size = 0
        if len(rad_picks) > 0:
            size += 1
        if len(tan_picks) > 0:
            size += 1
        if len(ver_picks) > 0:
            size += 1
        if len(abs_picks) > 0:
            size += 1

        if size == 2:
            plot_both = 1
            if len(rad_picks) == 0:
                fig, (ax2, ax3) = plt.subplots(ncols=2, figsize=(12, 5))
            if len(tan_picks) == 0:
                fig, (ax1, ax3) = plt.subplots(ncols=2, figsize=(12, 5))
            if len(ver_picks) == 0:
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        if size == 3:
            plot_both = 1
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))

        # if halve != False:
        # 	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 5))
        # else:
        # 	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        dx = 0.0
        dy = 0.0

        if len(rad_picks) > 0:
            if plot_both == 0:
                fig, ax1 = plt.subplots()
            x_min = min(xy_rad[:, 0])
            x_max = max(xy_rad[:, 0])
            y_min = min(xy_rad[:, 1])
            y_max = max(xy_rad[:, 1])
            nx, ny = 200, 200

            xi = np.linspace(x_min, x_max, nx)
            yi = np.linspace(y_min, y_max, ny)
            xi, yi = np.meshgrid(xi, yi)
            xi_rad = xi
            yi_rad = yi
            zi_rad = griddata((xy_rad[:, 0], xy_rad[:, 1]), mag_rad, (xi, yi), method='cubic')

            ax1.set(adjustable='box', aspect='equal')
            patch, xy_circle = mplo.cut_circle_patch(center, radius, ax1, 350, halve=halve)
            ax1.plot(xy_circle[:, 0], xy_circle[:, 1], '-k', linewidth=2)
            im11 = ax1.pcolormesh(xi_rad - dx, yi_rad + dx, zi_rad, cmap=plt.get_cmap('hot'))
            im21 = ax1.contour(xi_rad - dx, yi_rad + dx, zi_rad, colors="black")
            im = ax1.scatter(xy_rad[:, 0] - dx, xy_rad[:, 1] + dy, s=2, c="black")
            clb1 = fig.colorbar(im11, shrink=0.8, extend='both', ax=ax1)
            clb1.ax.tick_params(labelsize=30)
            clb1.ax.set_title('$B [\mathrm{fT}]$', fontsize=30)
            ax1.set_title("$\mathrm{radial}$", fontsize=30)
            ax1.axis('off')

        if len(tan_picks) > 0:
            if plot_both == 0:
                fig, ax2 = plt.subplots()
            x_min = min(xy_tan[:, 0])
            x_max = max(xy_tan[:, 0])
            y_min = min(xy_tan[:, 1])
            y_max = max(xy_tan[:, 1])
            nx, ny = 200, 200

            xi = np.linspace(x_min, x_max, nx)
            yi = np.linspace(y_min, y_max, ny)
            xi, yi = np.meshgrid(xi, yi)
            xi_tan = xi
            yi_tan = yi
            zi_tan = griddata((xy_tan[:, 0], xy_tan[:, 1]), mag_tan, (xi, yi), method='cubic')

            ax2.set(adjustable='box', aspect='equal')
            patch, xy_circle = mplo.cut_circle_patch(center, radius, ax2, 350, halve=halve)
            ax2.plot(xy_circle[:, 0], xy_circle[:, 1], '-k', linewidth=2)
            im12 = ax2.pcolormesh(xi_tan - dx, yi_tan + dx, zi_tan, cmap=plt.get_cmap('hot'))
            im22 = ax2.contour(xi_tan - dx, yi_tan + dx, zi_tan, colors="black")
            im = ax2.scatter(xy_tan[:, 0] - dx, xy_tan[:, 1] + dy, s=2, c="black")
            clb2 = fig.colorbar(im12, shrink=0.8, extend='both', ax=ax2)
            clb2.ax.set_title('$B [\mathrm{fT}]$', fontsize=30)
            clb2.ax.tick_params(labelsize=30)
            ax2.set_title("$\mathrm{tangential}$", fontsize=30)
            ax2.axis('off')

        if len(ver_picks) > 0:
            if plot_both == 0:
                fig, ax3 = plt.subplots()
            x_min = min(xy_ver[:, 0])
            x_max = max(xy_ver[:, 0])
            y_min = min(xy_ver[:, 1])
            y_max = max(xy_ver[:, 1])
            nx, ny = 200, 200

            xi = np.linspace(x_min, x_max, nx)
            yi = np.linspace(y_min, y_max, ny)
            xi, yi = np.meshgrid(xi, yi)
            xi_ver = xi
            yi_ver = yi
            zi_ver = griddata((xy_ver[:, 0], xy_ver[:, 1]), mag_ver, (xi, yi), method='cubic')

            ax3.set(adjustable='box', aspect='equal')
            patch, xy_circle = mplo.cut_circle_patch(center, radius, ax3, 350, halve=halve)
            ax3.plot(xy_circle[:, 0], xy_circle[:, 1], '-k', linewidth=2)
            im12 = ax3.pcolormesh(xi_ver - dx, yi_ver + dx, zi_ver, cmap=plt.get_cmap('hot'))
            im22 = ax3.contour(xi_ver - dx, yi_ver + dx, zi_ver, colors="black")
            im = ax3.scatter(xy_ver[:, 0] - dx, xy_ver[:, 1] + dy, s=2, c="black")
            clb2 = fig.colorbar(im12, shrink=0.8, extend='both', ax=ax3)
            clb2.ax.set_title('$B [\mathrm{fT}]$', fontsize=30)
            clb2.ax.tick_params(labelsize=30)
            ax3.set_title("$\mathrm{axial}$", fontsize=30)
            ax3.axis('off')

        if len(abs_picks) > 0:
            if plot_both == 0:
                fig, ax1 = plt.subplots()
            x_min = min(xy_abs[:, 0])
            x_max = max(xy_abs[:, 0])
            y_min = min(xy_abs[:, 1])
            y_max = max(xy_abs[:, 1])
            nx, ny = 200, 200

            xi = np.linspace(x_min, x_max, nx)
            yi = np.linspace(y_min, y_max, ny)
            xi, yi = np.meshgrid(xi, yi)
            xi_abs = xi
            yi_abs = yi
            zi_abs = griddata((xy_abs[:, 0], xy_abs[:, 1]), mag_abs, (xi, yi), method='cubic')

            ax1.set(adjustable='box', aspect='equal')
            patch, xy_circle = mplo.cut_circle_patch(center, radius, ax1, 350, halve=halve)
            ax1.plot(xy_circle[:, 0], xy_circle[:, 1], '-k', linewidth=2)
            im11 = ax1.pcolormesh(xi_abs - dx, yi_abs + dx, zi_abs, cmap=plt.get_cmap('hot'))
            im21 = ax1.contour(xi_abs - dx, yi_abs + dx, zi_abs, colors="black")
            im = ax1.scatter(xy_abs[:, 0] - dx, xy_abs[:, 1] + dy, s=2, c="black")
            clb1 = fig.colorbar(im11, shrink=0.8, extend='both', ax=ax1)
            clb1.ax.tick_params(labelsize=30)
            clb1.ax.set_title('$B [\mathrm{fT}]$', fontsize=30)
            # ax1.set_title("$\mathrm{"norm"}$", fontsize=30)
            ax1.axis('off')

        fig.tight_layout()
    #		plt.rc('font', family='serif')

    return fig


def plot_evokedobj_topo(evoked, data_path, block_name, system, time, position=None, multi=None, endalign=None):
    # implement to remove bad channels
    import mne
    import matplotlib.pyplot as plt
    import matplotlib.cbook as cbook
    import numpy as np
    import megtools.pyread_biosig as pbio
    import megtools.pymeg_visualize as pvis
    import megtools.vector_functions as vfun
    import megtools.meg_plot as mplo
    from scipy.interpolate import griddata
    import matplotlib.mlab as ml
    from matplotlib import rc
    rc('text', usetex=True)

    if system == "SQUID":
        lout_dict = {}
        lout = mne.channels.read_layout("ET160.lout", data_path + "data/")
        lout_names = lout.names
        lout_pos = lout.pos
        for i, j in enumerate(lout_names):
            lout_dict[j] = lout_pos[i]

        xy_all = []
        xy = []
        chs_names_evoked = evoked.ch_names
        for i, j in enumerate(lout_names):
            if j in chs_names_evoked:
                xy.append(lout_dict[j])
            xy_all.append(lout_dict[j])
        xy = np.array(xy)
        xy_all = np.array(xy_all)
        center = (np.mean(xy_all[:, 0]), np.mean(xy_all[:, 1]))
        radius = 1.08 * (np.max(xy_all[:, 0]) - np.min(xy_all[:, 0])) / 2.0
        no_ch_all = len(xy_all)
        no_ch = len(xy)

        mag_avg = evoked.data[:]
        if isinstance(time, list):
            mag_avg = np.average(mag_avg, axis=1)

        if position == "right":
            chosen = np.argsort(xy[:, 0])[::-1][0:int(0.6 * no_ch)]
            xy = xy[chosen, :]
            mag = mag_avg[chosen]
        elif position == "left":
            chosen = np.argsort(xy[:, 0])[0:int(0.6 * no_ch)]
            xy = xy[chosen, :]
            mag = mag_avg[chosen]
        else:
            chosen = []
            mag = mag_avg[:]

        chosen_time = np.argmin(np.abs(np.array(evoked.times) - time))
        mag = evoked.data[:, chosen_time]
        mag = mag * (10.0 ** 15.0)

        x_min = min(xy[:, 0])
        x_max = max(xy[:, 0])
        y_min = min(xy[:, 1])
        y_max = max(xy[:, 1])
        nx, ny = 200, 200

        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((xy[:, 0], xy[:, 1]), mag, (xi, yi), method='cubic')

        fig, ax = plt.subplots()
        im1 = plt.pcolormesh(xi, yi, zi, cmap=plt.get_cmap('hot'))
        im2 = plt.contour(xi, yi, zi, colors="black")
        im = plt.scatter(xy[:, 0], xy[:, 1], s=2, c="black")
        clb = fig.colorbar(im1, shrink=0.8, extend='both')
        clb.ax.set_title('$B [\mathrm{fT}]$', fontsize=30)
        clb.ax.tick_params(labelsize=30)
        patch, xy_circle = mplo.cut_circle_patch(center, radius, ax, 320)
        im1.set_clip_path(patch)
        plt.plot(xy_circle[:, 0], xy_circle[:, 1], '-k', linewidth=2)
        #		plt.tight_layout()
        plt.rc('font', family='serif')
        plt.axis('off')
    return fig


def create_squids(holders, directions, mag_type=0):
    import mne
    ch_types = len(holders) * ["mag"]
    ch_names = ["MEG " + '{:03}'.format(a) for a in np.arange(1, len(holders) + 1)]

    info = mne.create_info(ch_names=ch_names, sfreq=1, ch_types=ch_types)

    if mag_type == 0:
        coil_type = 6001
        kind = 1
        unit = 112
    if mag_type == 1:
        coil_type = 6002
        kind = 1
        unit = 112
    if mag_type == 2:
        coil_type = 2
        kind = 1
        unit = 112
    if mag_type == 3:
        coil_type = 6004
        kind = 1
        unit = 112

    unit_v = np.array([0.0, 0.0, 1.0])
    for i in range(len(ch_names)):
        theta = np.arctan2(np.sqrt(directions[i, 0] ** 2 + directions[i, 1] ** 2), directions[i, 2])
        phi = np.arctan2(directions[i, 1], directions[i, 0])

        import math
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

        angle0 = theta
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(angle0), -math.sin(angle0)],
                        [0, math.sin(angle0), math.cos(angle0)]])

        angle1 = np.pi / 2
        R_y = np.array([[math.cos(angle1), 0, math.sin(angle1)],
                        [0, 1, 0],
                        [-math.sin(angle1), 0, math.cos(angle1)]])

        angle2 = np.pi / 2 - phi
        R_z = np.array([[math.cos(angle2), -math.sin(angle2), 0],
                        [math.sin(angle2), math.cos(angle2), 0],
                        [0, 0, 1]])

        angle3 = -np.pi / 2
        R_z2 = np.array([[math.cos(angle3), -math.sin(angle3), 0],
                         [math.sin(angle3), math.cos(angle3), 0],
                         [0, 0, 1]])

        rot_mat = np.dot(R_z2, np.dot(R_x, R_z))

        info['chs'][i]['coil_type'] = coil_type
        info['chs'][i]['scanno'] = i + 1
        info['chs'][i]['logno'] = i + 1
        info['chs'][i]['kind'] = kind
        info['chs'][i]['range'] = 1.0
        info['chs'][i]['cal'] = 1.0
        info['chs'][i]['unit'] = unit
        info['chs'][i]['loc'] = np.array(
            [holders[i, 0], holders[i, 1], holders[i, 2], rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2],
             rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]])

    return info


def create_squids2(holders, rot_mat, mag_number=0):
    import mne
    ch_types = len(holders) * ["mag"]
    ch_names = ["MEG " + '{:03}'.format(a) for a in np.arange(1, len(holders) + 1)]

    info = mne.create_info(ch_names=ch_names, sfreq=1, ch_types=ch_types)

    kind = 1
    unit = 112

    for i in range(len(ch_names)):
        info['chs'][i]['coil_type'] = mag_number
        info['chs'][i]['scanno'] = i + 1
        info['chs'][i]['logno'] = i + 1
        info['chs'][i]['kind'] = kind
        info['chs'][i]['range'] = 1.0
        info['chs'][i]['cal'] = 1.0
        info['chs'][i]['unit'] = unit
        info['chs'][i]['loc'] = np.array(
            [holders[i, 0], holders[i, 1], holders[i, 2], rot_mat[i, 0, 0], rot_mat[i, 0, 1], rot_mat[i, 0, 2],
             rot_mat[i, 1, 0], rot_mat[i, 1, 1], rot_mat[i, 1, 2], rot_mat[i, 2, 0], rot_mat[i, 2, 1],
             rot_mat[i, 2, 2]])

    return info


def plot_magnetometers3(subject_dir, name, xyz1, rot_mat, magnetometer_number, coil_def, filename="rand_name"):
    # plot sensors
    import os
    import megtools.my3Dplot as m3p

    accuracy = 2

    surface1 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_inner_skull_surface')
    surface2 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skull_surface')
    surface3 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skin_surface')

    surfaces = [surface1, surface2, surface3]

    with open(coil_def) as f:
        content = f.readlines()

    for j, i in enumerate(content):
        if (i.split()[0].isdigit()):
            if magnetometer_number == int(i.split()[1]) and accuracy == int(i.split()[2]):
                magnetometer_gradiometer = int(i.split()[0])
                print(i, j)
                idx_range = [j+1, j+int(i.split()[3])+1]


    arr = []
    for i in range(idx_range[0], idx_range[1]):
        arr.append(content[i].split())

    arr = np.array(arr, dtype=float)

    uniques, counts = np.unique(arr[:, 0], return_counts=True)

    locations = xyz1[:, 0:3]
    locations2 = []
    directions2 = []
    unit_v = np.array([0., 0., 1.])
    directions = np.zeros(np.shape(locations))

    elements = []
    temp_list_ind = 0
    for i in range(len(locations)):
        for jj, ii in enumerate(uniques):
            temp_list_el = []
            for j in range(len(arr)):
                if arr[j, 0] == ii:
                    arr_temp = np.dot(arr[j, 1:4], rot_mat[i])
                    locations2.append(locations[i]+arr_temp)
                    directions2.append(np.dot(arr[j, 4:7], rot_mat[i]))
                    temp_list_el.append(temp_list_ind)
                    temp_list_ind += 1
            elements.append(temp_list_el)

            # print(ii)
        # for j in range(len(arr)):
        #     arr_temp = np.dot(arr[j, 1:4], rot_mat[i])
        #     locations2.append(locations[i]+arr_temp)
        #     directions2.append(np.dot(arr[j, 4:7], rot_mat[i]))

        # directions[i] = np.dot(unit_v, rot_mat[i])

    sensors = np.hstack((np.array(locations2), np.array(directions2)))

    p1 = plot_sensors_pyvista1(surfaces, sensors=sensors, elements=elements, arrow_color="black", grad=magnetometer_gradiometer)

    # p1.show(screenshot=name + 'opm_rad.png')
    # p1 = m3p.plot_sensors_pyvista(surfaces, sensors=[])
    p1.show(screenshot=filename+'.png')
    p1.show()
    p1.close()

    return


def plot_sensors_pyvista1(surfaces, sensors, sensors2=[], elements=[], arrow_color="black", grad=0):
    import pyvista as pv
    import mne
    import numpy as np
    import random

    pv.set_plot_theme("document")
    p = pv.Plotter()

    # for i in sensors:
    #     sphere = pv.Sphere(center=i[0:3] * 10 ** 3, radius=5)
    #     p.add_mesh(sphere, color="black")
    #     arrow = pv.Arrow(start=i[0:3] * 10 ** 3, direction=i[3:6], scale=15)
    #     p.add_mesh(arrow, color=arrow_color)

    for cunt, element in enumerate(elements):
            pts = np.zeros((len(element),3))
            if grad == 3 or grad == 2:
                if cunt % 2:
                    color = (random.uniform(0., 1.), random.uniform(0., 1.), random.uniform(0., 1.))
            else:
                color = "gray"
            for i, j in enumerate(element):
                pts[i,0] = sensors[j, 0]*10**3
                pts[i,1] = sensors[j, 1]*10**3
                pts[i,2] = sensors[j, 2]*10**3
            if len(element) == 1:
                sphere = pv.Sphere(center=pts, radius=1)
                p.add_mesh(sphere, color="black")
                arrow = pv.Arrow(start=pts[i, 0:3], direction=sensors[j, 3:6], scale=20)
                p.add_mesh(arrow, color=arrow_color)
            if len(element) == 4:
                faces = np.array([4, 0, 1, 2, 3])
                mesh = pv.PolyData(pts, faces)
                p.add_mesh(mesh, color=color)
            if len(element) == 6:
                faces = np.array([6, 0, 3, 5, 1, 4, 2])
                mesh = pv.PolyData(pts, faces)
                p.add_mesh(mesh, color=color)
            if len(element) == 8:
                faces = np.array([[4, 0, 1, 3, 2], [4, 4, 5, 7, 6], [4, 2, 3, 7, 6], [4, 0, 1, 5, 4], [4, 1, 3, 7, 5],
                                  [4, 0, 2, 6, 4]])
                mesh = pv.PolyData(pts, faces)
                p.add_mesh(mesh, color="red")
                arrow = pv.Arrow(start=np.mean(pts, axis=0), direction=sensors[j, 3:6], scale=20)
                p.add_mesh(arrow, color=arrow_color)

    step = 1.0 / (len(surfaces) + 1)
    opacities = np.linspace(1 - step, 0, num=len(surfaces), endpoint=False)
    for i, surface in enumerate(surfaces):
        rr_mm, tris = mne.read_surface(surface)
        tres = np.ones((len(tris), 1), dtype=int) * 3
        tris = np.hstack([tres, tris])
        gray1 = (0.5, 0.5, 0.5)
        polygon = pv.PolyData(rr_mm, tris)
        p.add_mesh(polygon, color=gray1, opacity=opacities[i] - 0.3)

    camera = pv.Camera()
    # p.camera.zoom(1.6)
    p.camera.zoom(3.0)

    return p


def plot_magnetometers2(subject_dir, name, xyz1, rot_mat, magnetometer_type):
    # plot sensors
    import os
    import megtools.my3Dplot as m3p
    surface1 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_inner_skull_surface')
    surface2 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skull_surface')
    surface3 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skin_surface')

    surfaces = [surface1, surface2, surface3]

    locations = xyz1[:, 0:3]
    unit_v = np.array([0., 0., 1.])
    directions = np.zeros(np.shape(locations))
    for i in range(len(locations)):
        directions[i] = np.dot(unit_v, rot_mat[i])

    sensors = np.hstack((locations, directions))

    p1 = m3p.plot_sensors_pyvista(surfaces, sensors=sensors, arrow_color="green")

    # p1.show(screenshot=name + 'opm_rad.png')
    # p1 = m3p.plot_sensors_pyvista(surfaces, sensors=[])
    # p1.show(screenshot='tan_components.png')

    p1.show()
    p1.close()

    return


def plot_magnetometers(subject_dir, name, xyz1, xyz2, magnetometer_type):
    # plot sensors
    import os
    import megtools.my3Dplot as m3p
    surface1 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_inner_skull_surface')
    surface2 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skull_surface')
    surface3 = os.path.join(subject_dir, name, 'bem', 'watershed', name + '_outer_skin_surface')

    surfaces = [surface1, surface2, surface3]
    if magnetometer_type in [1, 3, 4]:
        p1 = m3p.plot_sensors_pyvista(surfaces, sensors=xyz1[::], sensors2=xyz2[::], arrow_color="red")
    else:
        p1 = m3p.plot_sensors_pyvista(surfaces, sensors=xyz1[::], arrow_color="red")
    # p1.show(screenshot=name + 'opm_rad.png')
    # p1 = m3p.plot_sensors_pyvista(surfaces, sensors=[])
    # p1.show(screenshot='all_surfaces.png')
    p1.show()

    return


def create_opms(holders, components=["rad", "tan", "ver"]):
    import mne
    ch_types = int(len(holders) * (len(components) / 3)) * ["mag"]
    ch_numbers = []

    ch_numbers1 = []
    ch_numbers2 = []
    ch_numbers3 = []
    if "rad" in components:
        ch_numbers1 = range(0, len(holders), 3)
        ch_names1 = ["rad" + '{:03}'.format(a) for a in np.arange(int(len(holders) / 3))]
    if "tan" in components:
        ch_numbers2 = range(1, len(holders), 3)
        ch_names2 = ["tan" + '{:03}'.format(a) for a in np.arange(int(len(holders) / 3))]
    if "ver" in components:
        ch_numbers3 = range(2, len(holders), 3)
        ch_names3 = ["ver" + '{:03}'.format(a) for a in np.arange(int(len(holders) / 3))]

    # for i in range(len(list(ch_numbers1))):
    ch_numbers = list(ch_numbers1) + list(ch_numbers2) + list(ch_numbers3)
    ch_numbers = sorted([i for i in ch_numbers])

    ch_names = []
    ii_temp = int(len(ch_numbers) / len(components))
    print(ii_temp)
    for i in range(ii_temp):
        if "rad" in components:
            ch_names.append(ch_names1[i])
        if "tan" in components:
            ch_names.append(ch_names2[i])
        if "ver" in components:
            ch_names.append(ch_names3[i])

    # ch_names = [str(a) + str(b) for a, b in zip(ch_types, [str(x + 1) for x in range(len(holders) + 1)])]

    info = mne.create_info(ch_names=ch_names, sfreq=1, ch_types=ch_types)

    unit_v = np.array([0.0, 0.0, 1.0])
    for j, i in enumerate(ch_numbers):
        rot_mat = vfun.create_rot_matrix(holders[i, 3:6], unit_v)
        info['chs'][j]['coil_type'] = 9999
        info['chs'][j]['scanno'] = j + 1
        info['chs'][j]['logno'] = j + 1
        info['chs'][j]['kind'] = 1
        info['chs'][j]['range'] = 1.0
        info['chs'][j]['cal'] = 3.7000000285836165e-10
        info['chs'][j]['unit'] = 112
        info['chs'][j]['loc'] = np.array(
            [holders[i, 0], holders[i, 1], holders[i, 2], rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2],
             rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]])

    return info


def import_sensor_pos_ori_all(sensorholder_path, name, subject_dir, gen12=1):
    import megtools.vector_functions as vfun
    import mne

    if gen12 == 2:
        sensorholder_file = sensorholder_path + name + '_sensor_pos_ori_all_v2.txt'
        opm_trans_path = sensorholder_path + "opm_trans_v2.txt"
    else:
        sensorholder_file = sensorholder_path + name + '_sensor_pos_ori_all.txt'
        opm_trans_path = sensorholder_path + "opm_trans.txt"

    rotation, translation = pbio.import_opm_trans(opm_trans_path, name)
    translation = translation / 1000.0

    holders = pbio.imp_sensor_holders(sensorholder_file)
    holders[:, 0:3] = holders[:, 0:3] / 1000.0

    if gen12 == 2:
        # !!!!! be very carefull what comes first.
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

    holders[:, 1], holders[:, 2] = vfun.rotate_via_numpy(holders[:, 1], holders[:, 2], np.radians(-rotation[0]))
    holders[:, 0], holders[:, 2] = vfun.rotate_via_numpy(holders[:, 0], holders[:, 2], np.radians(-rotation[1]))
    holders[:, 0], holders[:, 1] = vfun.rotate_via_numpy(holders[:, 0], holders[:, 1], np.radians(-rotation[2]))
    holders[:, 4], holders[:, 5] = vfun.rotate_via_numpy(holders[:, 4], holders[:, 5], np.radians(-rotation[0]))
    holders[:, 3], holders[:, 5] = vfun.rotate_via_numpy(holders[:, 3], holders[:, 5], np.radians(-rotation[1]))
    holders[:, 3], holders[:, 4] = vfun.rotate_via_numpy(holders[:, 3], holders[:, 4], np.radians(-rotation[2]))

    if gen12 != 2:
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

        surf = mne.read_surface(subject_dir + name + "/surf/" + "lh.white", read_metadata=True)
        holders[:, 0:3] = holders[:, 0:3] - (surf[2]['cras'] / 1000.0)

    return holders


def get_tangential_directions(xyz1, xyz2, plane="xy"):
    import numpy as np
    import random
    from math import cos, sin, pi
    from numpy import linalg as LA
    import megtools.vector_functions as vfun

    xyz1_n = np.copy(xyz1)
    xyz2_n = np.copy(xyz2)
    xyz1_c = np.copy(xyz1)
    xyz2_c = np.copy(xyz2)

    for i in range(len(xyz1)):
        dir1 = xyz1[i, 3:6]
        dir1[2] = 0
        dir1 = dir1 / (LA.norm(dir1))
        dir2 = np.empty(3)
        dir2[1] = -(dir1[0] / dir1[1]) * dir1[0]
        dir2[0] = dir1[0]
        dir2[2] = 0
        dir2 = dir2 / (LA.norm(dir2))
        if np.cross(dir2, dir1)[2] > 0:
            dir2 = -dir2
        dir2 = dir2 / (LA.norm(dir2))
        xyz1_n[i, 3:6] = dir2
        xyz2_n[i, 3:6] = dir2

    if plane == "xz" or plane == "yz" or plane == "xyz":
        for i in range(len(xyz1)):
            dir1 = np.copy(xyz1_c[i, 3:6])
            dir1 = dir1 / (LA.norm(dir1))
            dir2 = np.copy(xyz1_n[i, 3:6])
            dir2 = dir2 / (LA.norm(dir2))
            dir3 = np.cross(dir1, dir2)
            dir3 = dir3 / (LA.norm(dir3))
            xyz1_n[i, 3:6] = dir3
            xyz2_n[i, 3:6] = dir3

    return xyz1_n, xyz2_n


def get_planar_gradiometers(xyz1, xyz2, plane="xy"):
    import numpy as np
    import random
    from math import cos, sin, pi
    from numpy import linalg as LA
    import megtools.vector_functions as vfun

    xyz1_n = np.copy(xyz1)
    xyz2_n = np.copy(xyz2)
    xyz1_c = np.copy(xyz1)
    xyz2_c = np.copy(xyz2)

    xyz_m = (xyz1[:, 0:3] + xyz2[:, 0:3]) / 2.0
    dist = np.sqrt((xyz1[:, 0] - xyz2[:, 0]) ** 2 + (xyz1[:, 1] - xyz2[:, 1]) ** 2 + (xyz1[:, 2] - xyz2[:, 2]) ** 2)

    for i in range(len(xyz1)):
        dir1 = xyz1[i, 3:6]
        dir1[2] = 0
        dir1 = dir1 / (LA.norm(dir1))
        dir2 = np.empty(3)
        dir2[1] = -(dir1[0] / dir1[1]) * dir1[0]
        dir2[0] = dir1[0]
        dir2[2] = 0
        dir2 = dir2 / (LA.norm(dir2))
        if np.cross(dir2, dir1)[2] < 0:
            dir2 = -dir2
        dir2 = dir2 / (LA.norm(dir2))

        if plane == "xz" or plane == "yz" or plane == "xyz":
            dir2 = np.cross(dir1, dir2)

        xyz1_n[i, 0:3] = xyz1[i, 0:3] + 0.1 * dist[i] * dir2
        xyz2_n[i, 0:3] = xyz1[i, 0:3] - 0.1 * dist[i] * dir2

    return xyz1_n, xyz2_n


def import_sensor_pos_squid(template_path, fwd_squid_path, fname_trans_squid, tangential=0, planar_gradiometer=0,
                            grad_plane="xy", dir_plane="xy"):
    # dir_plane "xy", "yz", "xyz"
    import mne
    xyz1, xyz2, ch_info, sfreq = pbio.import_sensors(template_path, "squid")
    fwd_squid = mne.read_forward_solution(fwd_squid_path)

    xyz2[:, 0] = -xyz2[:, 0]
    temp = np.copy(xyz2[:, 1])
    xyz2[:, 1] = xyz2[:, 2]
    xyz2[:, 2] = temp
    xyz2[:, 3] = -xyz2[:, 3]
    temp = np.copy(xyz2[:, 4])
    xyz2[:, 4] = xyz2[:, 5]
    xyz2[:, 5] = temp

    xyz1[:, 0] = -xyz1[:, 0]
    temp = np.copy(xyz1[:, 1])
    xyz1[:, 1] = xyz1[:, 2]
    xyz1[:, 2] = temp
    xyz1[:, 3] = -xyz1[:, 3]
    temp = np.copy(xyz1[:, 4])
    xyz1[:, 4] = xyz1[:, 5]
    xyz1[:, 5] = temp

    tan_directions = len(xyz1) * [0., 1., 0.]

    if planar_gradiometer == 1:
        xyz1, xyz2 = get_planar_gradiometers(xyz1, xyz2, plane=grad_plane)

    if tangential == 1:
        xyz1, xyz2 = get_tangential_directions(xyz1, xyz2, plane=dir_plane)

    xyz1_squid = xyz1
    xyz2_squid = xyz2

    dev_head_t = fwd_squid['info']['dev_head_t']
    head_mri_t = mne.read_trans(fname_trans_squid)

    xyz1_squid[:, 0:3] = mne.transforms.apply_trans(dev_head_t, xyz1_squid[:, 0:3])
    xyz1_squid[:, 0:3] = mne.transforms.apply_trans(head_mri_t, xyz1_squid[:, 0:3])
    xyz1_squid[:, 3:6] = np.dot(dev_head_t["trans"][0:3, 0:3], xyz1_squid[:, 3:6].T).T
    xyz1_squid[:, 3:6] = np.dot(head_mri_t["trans"][0:3, 0:3], xyz1_squid[:, 3:6].T).T

    xyz2_squid[:, 0:3] = mne.transforms.apply_trans(dev_head_t, xyz2_squid[:, 0:3])
    xyz2_squid[:, 0:3] = mne.transforms.apply_trans(head_mri_t, xyz2_squid[:, 0:3])
    xyz2_squid[:, 3:6] = np.dot(dev_head_t["trans"][0:3, 0:3], xyz2_squid[:, 3:6].T).T
    xyz2_squid[:, 3:6] = np.dot(head_mri_t["trans"][0:3, 0:3], xyz2_squid[:, 3:6].T).T

    return xyz1_squid, xyz2_squid


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
			angle_between((1, 0, 0), (0, 1, 0))
			1.5707963267948966
			angle_between((1, 0, 0), (1, 0, 0))
			0.0
			angle_between((1, 0, 0), (-1, 0, 0))
			3.141592653589793
	"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def import_sensor_pos_squid2(template_path, fwd_squid_path, fname_trans_squid, name, tangential=0, planar_gradiometer=0,
                             grad_plane="xy", dir_plane="xy"):
    # dir_plane "xy", "yz", "xyz"
    import numpy as np
    import mne
    xyz1, xyz2, ch_info, sfreq = pbio.import_sensors(template_path, "squid")
    fwd_squid = mne.read_forward_solution(fwd_squid_path)

    xyz2[:, 0] = -xyz2[:, 0]
    temp = np.copy(xyz2[:, 1])
    xyz2[:, 1] = xyz2[:, 2]
    xyz2[:, 2] = temp
    xyz2[:, 3] = -xyz2[:, 3]
    temp = np.copy(xyz2[:, 4])
    xyz2[:, 4] = xyz2[:, 5]
    xyz2[:, 5] = temp

    xyz1[:, 0] = -xyz1[:, 0]
    temp = np.copy(xyz1[:, 1])
    xyz1[:, 1] = xyz1[:, 2]
    xyz1[:, 2] = temp
    xyz1[:, 3] = -xyz1[:, 3]
    temp = np.copy(xyz1[:, 4])
    xyz1[:, 4] = xyz1[:, 5]
    xyz1[:, 5] = temp

    xyz1_orig = np.copy(xyz1)

    # GET SENSOR ROTATIONS
    if planar_gradiometer == 1:
        xyz1, xyz2 = get_planar_gradiometers(xyz1, xyz2, plane=grad_plane)

    if tangential == 1:
        xyz1, xyz2 = get_tangential_directions(xyz1, xyz2, plane=dir_plane)

    xyz1_temp, xyz2_temp = get_planar_gradiometers(np.copy(xyz1), np.copy(xyz2), plane="xy")

    xyz1_squid = np.copy(xyz1)
    xyz2_squid = np.copy(xyz2)

    dev_head_t = fwd_squid['info']['dev_head_t']
    head_mri_t = mne.read_trans(fname_trans_squid)

    xyz1_squid[:, 0:3] = mne.transforms.apply_trans(dev_head_t, xyz1_squid[:, 0:3])
    xyz1_squid[:, 0:3] = mne.transforms.apply_trans(head_mri_t, xyz1_squid[:, 0:3])
    xyz1_squid[:, 3:6] = np.dot(dev_head_t["trans"][0:3, 0:3], xyz1_squid[:, 3:6].T).T
    xyz1_squid[:, 3:6] = np.dot(head_mri_t["trans"][0:3, 0:3], xyz1_squid[:, 3:6].T).T

    xyz2_squid[:, 0:3] = mne.transforms.apply_trans(dev_head_t, xyz2_squid[:, 0:3])
    xyz2_squid[:, 0:3] = mne.transforms.apply_trans(head_mri_t, xyz2_squid[:, 0:3])
    xyz2_squid[:, 3:6] = np.dot(dev_head_t["trans"][0:3, 0:3], xyz2_squid[:, 3:6].T).T
    xyz2_squid[:, 3:6] = np.dot(head_mri_t["trans"][0:3, 0:3], xyz2_squid[:, 3:6].T).T

    xyz1_temp[:, 0:3] = mne.transforms.apply_trans(dev_head_t, xyz1_temp[:, 0:3])
    xyz1_temp[:, 0:3] = mne.transforms.apply_trans(head_mri_t, xyz1_temp[:, 0:3])
    xyz1_temp[:, 3:6] = np.dot(dev_head_t["trans"][0:3, 0:3], xyz1_temp[:, 3:6].T).T
    xyz1_temp[:, 3:6] = np.dot(head_mri_t["trans"][0:3, 0:3], xyz1_temp[:, 3:6].T).T

    xyz2_temp[:, 0:3] = mne.transforms.apply_trans(dev_head_t, xyz2_temp[:, 0:3])
    xyz2_temp[:, 0:3] = mne.transforms.apply_trans(head_mri_t, xyz2_temp[:, 0:3])
    xyz2_temp[:, 3:6] = np.dot(dev_head_t["trans"][0:3, 0:3], xyz2_temp[:, 3:6].T).T
    xyz2_temp[:, 3:6] = np.dot(head_mri_t["trans"][0:3, 0:3], xyz2_temp[:, 3:6].T).T

    sen_rotation = xyz2_temp[:, 0:3] - xyz1_temp[:, 0:3]
    for i in range(len(sen_rotation)):
        sen_rotation[i] = unit_vector(sen_rotation[i])

    def get_rotation_matrix(xyz1_orig, sen_rotation, rot_plane):
        unit_v = np.array([0.0, 1.0, 0.0])
        rot_mat = np.zeros([len(xyz1_orig), 3, 3])
        for i in range(len(xyz1_orig)):
            theta = np.arctan2(np.sqrt(xyz1_orig[i, 3] ** 2 + xyz1_orig[i, 4] ** 2), xyz1_orig[i, 5])
            phi = np.arctan2(xyz1_orig[i, 3], xyz1_orig[i, 4])

            import math
            R = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

            angle0 = theta
            R_x = np.array([[1, 0, 0],
                            [0, math.cos(angle0), -math.sin(angle0)],
                            [0, math.sin(angle0), math.cos(angle0)]])

            # angle1=np.pi/2
            # R_y = np.array([[math.cos(angle1), 0, math.sin(angle1)],
            # 				[0, 1, 0],
            # 				[-math.sin(angle1), 0, math.cos(angle1)]])

            angle2 = phi
            R_z = np.array([[math.cos(angle2), -math.sin(angle2), 0],
                            [math.sin(angle2), math.cos(angle2), 0],
                            [0, 0, 1]])

            # plane_normal = np.array([0.0, 0.0, 1.0])
            vector = [1.0, 0.0, 0.0]
            sen_rotation[i] = np.dot(sen_rotation[i], np.linalg.inv(np.dot(R_x, R_z)))
            alpha = angle_between(sen_rotation[i], vector)

            if rot_plane == "yz":
                alpha = -np.pi / 2 - alpha

            angle3 = alpha

            R_z2 = np.array([[math.cos(angle3), -math.sin(angle3), 0],
                             [math.sin(angle3), math.cos(angle3), 0],
                             [0, 0, 1]])
            # R_x2 = np.array([[1, 0, 0],
            # 				[0, math.cos(angle3), -math.sin(angle3)],
            # 				[0, math.sin(angle3), math.cos(angle3)]])

            rot_mat[i] = np.dot(R_z2, np.dot(R_x, R_z))
        # rot_mat[i] = np.dot(R_x, R_z)
        return rot_mat

    # for i in range(len(xyz1_squid)):
    # 	VP1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    # 	VP2 = np.array([[0.0, 0.0, 0.0], xyz1_squid[i,3:6], sen_rotation[i]])
    # mat, trans =  vfun.rigid_transform_3D(VP1, VP2)

    rotation_matrix = get_rotation_matrix(np.copy(xyz1_squid), sen_rotation, rot_plane=grad_plane)

    return xyz1_squid, xyz2_squid, rotation_matrix


class AvgStatistics:
    def __init__(self):
        import numpy as np

        self.dist = []  # Array of all distances
        self.cc = []  # Array of all ccs
        self.re = []  # Array of all res
        self.snr = []  # Array of all snrs
        self.snrdb = []  # Array of all snrdbs

        self.names = np.array(())  # Array of all names

        self.avgdist = []
        self.avgcc = []
        self.avgre = []
        self.avgsnr = []
        self.avgsnrdb = []
        self.noisestr = []
        self.spontnoise = []

        self.avg_avgdist = 0
        self.avg_avgcc = 0
        self.avg_avgre = 0
        self.avg_snr = 0
        self.avg_snrdb = 0

        self.avg_avgdist2 = 0
        self.avg_avgcc2 = 0
        self.avg_avgre2 = 0
        self.avg_snr2 = 0
        self.avg_snrdb2 = 0

        self.std_avgdist = 0
        self.std_avgcc = 0
        self.std_avgre = 0
        self.std_snr = 0
        self.std_snrdb = 0

        self.std_avgdist2 = 0
        self.std_avgcc2 = 0
        self.std_avgre2 = 0
        self.std_snr2 = 0
        self.std_snrdb2 = 0

    def add_all_names(self, names):
        self.names = names

    def add_name(self, name):
        self.dist.append([])
        self.cc.append([])
        self.re.append([])
        self.snr.append([])
        self.snrdb.append([])

        self.avgdist.append([])
        self.avgcc.append([])
        self.avgre.append([])
        self.avgsnr.append([])
        self.avgsnrdb.append([])
        self.noisestr.append([])
        self.spontnoise.append([])

        if len(self.names) == 0:
            self.names = [name]
        else:
            self.names.append(name)

    def add_noisestr(self, values_arr, subj_i):
        self.noisestr[subj_i].append(values_arr)

    def add_spontnoise(self, values_arr, subj_i):
        self.spontnoise[subj_i].append(values_arr)

    def add_dists(self, values_arr, subj_i):
        self.dist[subj_i].append(values_arr)

    def add_ccs(self, values_arr, subj_i):
        self.cc[subj_i].append(values_arr)

    def add_res(self, values_arr, subj_i):
        self.re[subj_i].append(values_arr)

    def add_snrs(self, values_arr, subj_i):
        self.snr[subj_i].append(values_arr)

    def add_snr_dbs(self, values_arr, subj_i):
        self.snrdb[subj_i].append(values_arr)

    def add_avgdist(self, values_arr, subj_i):
        self.avgdist[subj_i].append(values_arr)

    def add_avgcc(self, values_arr, subj_i):
        self.avgcc[subj_i].append(values_arr)

    def add_avgre(self, values_arr, subj_i):
        self.avgre[subj_i].append(values_arr)

    def add_avgsnr(self, values_arr, subj_i):
        self.avgsnr[subj_i].append(values_arr)

    def add_avgsnrdb(self, values_arr, subj_i):
        self.avgsnrdb[subj_i].append(values_arr)

    def calc_avgs(self):
        self.avg_avgdist = np.average(np.array(self.avgdist), axis=0)
        self.avg_avgcc = np.average(np.array(self.avgcc), axis=0)
        self.avg_avgre = np.average(np.array(self.avgre), axis=0)
        self.avg_snr = np.average(np.array(self.avgsnr), axis=0)
        self.avg_snrdb = np.average(np.array(self.avgsnrdb), axis=0)

    def calc_avgs2(self):
        self.avg_avgdist2 = np.average(self.dist, axis=(0, 2))
        self.avg_avgcc2 = np.average(self.cc, axis=(0, 2))
        self.avg_avgre2 = np.average(self.re, axis=(0, 2))
        self.avg_snr2 = np.average(self.snr, axis=(0, 2))
        self.avg_snrdb2 = np.average(self.snrdb, axis=(0, 2))

    def calc_stds(self):
        self.std_avgdist = np.std(np.array(self.avgdist), axis=0)
        self.std_avgcc = np.std(np.array(self.avgcc), axis=0)
        self.std_avgre = np.std(np.array(self.avgre), axis=0)
        self.std_snr = np.std(np.array(self.avgsnr), axis=0)
        self.std_snrdb = np.std(np.array(self.avgsnrdb), axis=0)

    def calc_stds2(self):
        self.std_avgdist2 = np.std(self.dist, axis=(0, 2))
        self.std_avgcc2 = np.std(self.cc, axis=(0, 2))
        self.std_avgre2 = np.std(self.re, axis=(0, 2))
        self.std_snr2 = np.std(self.snr, axis=(0, 2))
        self.std_snrdb2 = np.std(self.snrdb, axis=(0, 2))

    def save_obj(self, file_path):
        import pickle
        with open(file_path, 'wb') as output:
            pickle.dump(self, output)


class EvokedMaps:
    def __init__(self):
        import numpy as np
        self.data = np.array(())
        self.times = np.array(())
        self.xyz = np.array(())
        self.times = np.array(())
        self.names = np.array(())

    def add_names(self, names):
        self.names = names

    def add_channels(self, xyz):
        self.xyz = xyz

    def add_times(self, times):
        self.times = times

    def add_data(self, data):
        self.data = data

    def add_mne_evoked(self, evoked):
        import numpy as np
        if self.data.size == 0:
            self.data = evoked.data
            self.times = evoked.times
        else:
            self.data = np.hstack((self.data, evoked.data))
            self.times = np.hstack((self.times, evoked.times))
        channels = []
        for i in range(len(evoked.info['ch_names'])):
            channels.append(evoked.info["chs"][i]['loc'])
        channels = np.array(channels)
        self.xyz = channels
        self.names = evoked.ch_names

    def copy(self):
        import copy
        copied_evoked = copy.deepcopy(self)
        return copied_evoked

    def save_obj(self, file_path):
        import pickle
        with open(file_path, 'wb') as output:
            pickle.dump(self, output)

    def use_components(self, component):
        if type(component) == int:
            picks = np.arange(component, len(self.xyz), 3)
        elif len(component) == 1:
            picks = np.arange(component[0], len(self.xyz), 3)
        elif len(component) == 2:
            picks1 = np.arange(component[0], len(self.xyz), 3)
            picks2 = np.arange(component[1], len(self.xyz), 3)
            picks = np.hstack((picks1, picks2))
        else:
            picks = np.arange(0, len(self.xyz), 1)

        self.xyz = self.xyz[picks]
        self.data = self.data[picks, :]
        self.names = np.array(self.names)
        self.names = self.names[picks]
        self.names = self.names.tolist()


def read_obj(file_path):
    import pickle
    with open(file_path, 'rb') as input:
        self = pickle.load(input)
    return self


def plot_one_var(x, y, yerr=[], xname="", yname="", show=False, xscale=1, yscale=1, savefig=False, label_name=None,
                 legend=False):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    # fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(6)
    fig.set_figheight(4)
    x = xscale * np.array(x)
    y = yscale * np.array(y)

    ax.plot(x, y, linestyle='-', linewidth=2, c="black", markersize=5, marker="o", label=label_name)
    if legend:
        ax.legend()
    if len(yerr) > 0:
        yerr_temp = [yscale * i for i in yerr]
        ax.errorbar(x, y, yerr=yerr_temp, c="black", linestyle='none')

    ax.set_ylabel(yname, fontsize=15)
    ax.set_xlabel(xname, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    if isinstance(savefig, str):
        plt.savefig(savefig)

    if show == True:
        plt.show()

    return fig, ax


def add_one_var_fig(fig, ax, x, y, yerr=[], show=False, xscale=1, yscale=1, savefig=False, color="black",
                    label_name=None, legend=False):
    import matplotlib.pyplot as plt

    x = float(xscale) * np.array(x)
    y = yscale * y
    ax.plot(x, y, linestyle='-', linewidth=2, c=color, markersize=5, marker="o", label=label_name)

    if len(yerr) > 0:
        yerr_temp = [yscale * i for i in yerr]
        ax.errorbar(x, y, yerr=yerr_temp, c=color, linestyle='none')

    if legend:
        ax.legend()

    if show == True:
        fig.show()

    if isinstance(savefig, str):
        fig.savefig(savefig)

    return fig


if __name__ == '__main__':
    main()
