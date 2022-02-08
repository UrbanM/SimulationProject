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

    # FIGURE 5
    # visualize4("simulate3007,1908-all.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #            fname3="simulate3007,1908-ver.obj", labels=["all combined", "radial", "tangential (longitude)", "tangential (latitude)"])
    # visualize4_SNR("simulate2808-snr-all.obj", fname1="simulate2808-snr-rad.obj", fname2="simulate2808-snr-tan.obj",
    #            fname3="simulate2808-snr-ver.obj", labels=["all combined", "radial", "tangential (longitude)", "tangential (latitude)"])

    # FIGURE 6
    # visualize5("simulate3007,1908-all.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #            fname3="simulate3007,1908-ver.obj", labels=["all combined", "radial", "tangential (longitude)", "tangential (latitude)"])
    # visualize5_SNR("simulate2808-snr-all.obj", fname1="simulate2808-snr-rad.obj", fname2="simulate2808-snr-tan.obj",
    #            fname3="simulate2808-snr-ver.obj", labels=["all combined", "radial", "tangential (longitude)", "tangential (latitude)"])


    # FIGURE 7, 8, 9, 10
    # plot_contour("Archive/simulate3007,1908-all.obj", fname1="Archive/simulate3007,1908-rad.obj", fname2="Archive/simulate3007,1908-tan.obj",
    #             fname3="Archive/simulate3007,1908-ver.obj", labels=["OPM-ALL", "OPM-NOR", "OPM-TAN-LAT", "OPM-TAN-LON"], parameter="dist")
    # plot_contour("simulate3007,1908-all.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #             fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"], parameter="re")
    # plot_contour("simulate3007,1908-all.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #             fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"], parameter="cc")
    # plot_contour("simulate3007,1908-all.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #             fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"], parameter="snr")
    # plot_contour_SNR("simulate2808-snr-all.obj", fname1="simulate2808-snr-rad.obj", fname2="simulate2808-snr-tan.obj",
    #             fname3="simulate2808-snr-ver.obj", labels=["OPM-ALL", "OPM-NOR", "OPM-TAN-LAT", "OPM-TAN-LON"], parameter="snr")

    # FIGURE 11
    # visualize("simulate1608,1908-squid1.obj", fname1="simulate1608,1908-squid2.obj", fname2="simulate1608,1908-squid3.obj",
    #           fname3="simulate1608,1908-squid4.obj", fname4="simulate1608,1908-squid5.obj",
    #           fname5="simulate1608,1908-squid6.obj", labels=["axial gradiometer", "magnetometer (rad.)",
    #           "planar gradiometer (lat.)", "planar gradiometer (long.)", "magnetometer (tan. lat.)",
    #                                                          "magnetometer (tan. long.)"])
    # visualize_SNR("simulate2808-squid1-std.obj", fname1="simulate2808-squid2-std.obj", fname2="simulate2808-squid3-std.obj",
    #           fname3="simulate2808-squid4-std.obj", fname4="simulate2808-squid5-std.obj",
    #           fname5="simulate2808-squid6-std.obj", labels=["axial gradiometer", "magnetometer (rad.)",
    #           "planar gradiometer (lat.)", "planar gradiometer (long.)", "magnetometer (tan. lat.)",
    #                                                          "magnetometer (tan. long.)"])

    # FIGURE 12
    # visualize2("simulate1708-squid1.obj", fname1="simulate1708-squid2.obj", fname2="simulate1708-squid3.obj",
    #           fname3="simulate1708-squid4.obj", fname4="simulate1708-squid5.obj",
    #           fname5="simulate1708-squid6.obj", labels=["axial gradiometer", "magnetometer (rad.)",
    #           "planar gradiometer (lat.)", "planar gradiometer (long.)", "magnetometer (tan. lat.)",
    #                                                     "magnetometer (tan. long.)"])
    # visualize2_SNR("simulate2808-squid1-spont.obj", fname1="simulate2808-squid2-spont.obj",
    #                fname2="simulate2808-squid3-spont.obj", fname3="simulate2808-squid4-spont.obj",
    #                fname4="simulate2808-squid5-spont.obj", fname5="simulate2808-squid6-spont.obj",
    #                labels=["axial gradiometer", "magnetometer (rad.)", "planar gradiometer (lat.)",
    #                        "planar gradiometer (long.)", "magnetometer (tan. lat.)", "magnetometer (tan. long.)"])

    # FIGURE 13
    # visualize3("simulate2708-all.obj", fname1="simulate2708-rad.obj", fname2="simulate2708-tan.obj",
    #            fname3="simulate2708-ver.obj", labels=["all combined", "radial", "tangential (longitude)", "tangential (latitude)"])

    # FIGURE 14
    # visualize3("simulate2708-squid1.obj", fname1="simulate2708-squid2.obj", fname2="simulate2708-squid3.obj",
    #           fname3="simulate2708-squid4.obj", fname4="simulate2708-squid5.obj",
    #           fname5="simulate2708-squid6.obj", labels=["axial gradiometer", "magnetometer (rad.)",
    #           "planar gradiometer (lat.)", "planar gradiometer (long.)", "magnetometer (tan. lat.)",
    #                                                          "magnetometer (tan. long.)"])

    # FIGURE 15!!!!!!!
    # visualize_relative("simulate2608-all-relative.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #           fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"])
    # visualize_relative("Archive/simulate3008-all-relative.obj", fname1="Archive/simulate3007,1908-rad.obj", fname2="Archive/simulate3007,1908-tan.obj",
    #           fname3="Archive/simulate3007,1908-ver.obj", labels=["OPM-ALL", "OPM-NOR", "OPM-TAN-LAT", "OPM-TAN-LON"])
    # visualize_relative("simulate2808-all-relative.obj", fname1="simulate3007,1908-rad.obj", fname2="simulate3007,1908-tan.obj",
    #           fname3="simulate3007,1908-ver.obj", labels=["all", "rad", "tan", "ver"])
    # visualize_relative("Archive/simulate2908-all-relative.obj", fname1="Archive/simulate3007,1908-rad.obj", fname2="Archive/simulate3007,1908-tan.obj",
    #           fname3="Archive/simulate3007,1908-ver.obj", labels=["all combined", "radial", "tangential (longitude)", "tangential (latitude)"])

    # FIGURE 16
    # visualize3("simulate2708-squid2.obj", fname1="simulate2708-squid5.obj", fname2="simulate2708-squid6.obj",
    #           fname3="simulate2708-rad.obj", fname4="simulate2708-tan.obj", fname5="simulate2708-ver.obj",
    #           labels=["SQUID magnetometer (rad.)", "SQUID magnetometer (tan. lat.)", "SQUID magnetometer (tan. long.)",
    #           "OPM radial", "OPM tangential (latitude)", "OPM tangential (longitude)"], addtoname="-combined")

    # FIGURE 17 (NEW FIG6)
    # visualize4("Archive/simulate1608,1908-squid2.obj", fname1="Archive/simulate1608,1908-squid5.obj", fname2="Archive/simulate1608,1908-squid6.obj",
    #           fname3="Archive/simulate3007,1908-rad.obj", fname4="Archive/simulate3007,1908-tan.obj", fname5="Archive/simulate3007,1908-ver.obj",
    #           fname6="Archive/simulate3007,1908-all.obj",
    #           labels=["SQUID-NOR", "SQUID-TAN-LAT", "SQUID-TAN-LON",
    #           "OPM-NOR", "OPM-TAN-LAT", "OPM-TAN-LON", "OPM-ALL"], addtoname="-combined-new")

    #FIGURE 18 (NEW FIG5)
    # visualize5("simulate1708-squid2.obj", fname1="simulate1708-squid5.obj", fname2="simulate1708-squid6.obj",
    #           fname3="Archive/simulate3007,1908-rad.obj", fname4="Archive/simulate3007,1908-tan.obj",
    #           fname5="Archive/simulate3007,1908-ver.obj", fname6="Archive/simulate3007,1908-all.obj",
    #           labels=["SQUID-NOR", "SQUID-TAN-LAT", "SQUID-TAN-LON",
    #           "OPM-NOR", "OPM-TAN-LAT", "OPM-TAN-LON", "OPM-ALL"])

    # FIGURE reduced sensor count (NEW FIG9)
    # figure_reduced_sens(name="simulate0301")

    # FIGURE reduced sensor count (NEW FIG9B)
    figure_reduced_sens(name="simulate2601")

    return

def figure_reduced_sens(name):
    import matplotlib.pyplot as plt
    yattr = "avg_avgdist"
    xattr = "sensno"
    yerrattr = "std_avgdist"
    xscale = 10.0**2
    yscale = 10.0 ** 3
    yname = "$d_{\mathrm{s,f}} \mathrm{[mm]}$"

    # yattr = "avg_avgcc"
    # xattr = "sensno"
    # yerrattr = "std_avgcc"
    # xscale = 10.0**2
    # yscale = 1
    # yname = "CC"

    fig, ax = visualize_uniform(name + "-rad.obj", yattr, xattr, yerr_attr=yerrattr, label="OPM-NOR",
                                yname=yname, yscale=yscale, xscale=xscale,
                                xname="\% of sensors removed", marker="X")
    visualize_uniform(name + "-tan.obj", yattr, xattr, yerr_attr=yerrattr, fig=fig, ax=ax,
                      color="r", label="OPM-TAN-LAT", yscale=yscale, xscale=xscale, marker="s")
    visualize_uniform(name + "-ver.obj", yattr, xattr, yerr_attr=yerrattr, fig=fig, ax=ax,
                      color="g", label="OPM-TAN-LON", yscale=yscale, xscale=xscale, marker="o")
    visualize_uniform(name + "-radtanver.obj", yattr, xattr, yerr_attr=yerrattr, fig=fig, ax=ax,
                      color="b", label="OPM-ALL", yscale=yscale, xscale=xscale, marker="^")

    visualize_uniform(name + "-radtan.obj", yattr, xattr, yerr_attr=yerrattr, fig=fig, ax=ax,
                      color="orange", label="OPM-NOR,TAN-LAT", yscale=yscale, xscale=xscale, marker="P")
    visualize_uniform(name + "-radver.obj", yattr, xattr, yerr_attr=yerrattr, fig=fig, ax=ax,
                      color="purple", label="OPM-NOR,TAN-LON", yscale=yscale, xscale=xscale, marker="D")
    visualize_uniform(name + "-tanver.obj", yattr, xattr, yerr_attr=yerrattr, fig=fig, ax=ax,
                      color="brown", label="OPM-TAN-LAT,LON", yscale=yscale, xscale=xscale, marker="*")
    plt.savefig(name+"-reducedsens.png", dpi=300)
    plt.show()


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
    relative_x = 7.5 * 10 ** (-14.0)
    xscale = 1
    ratios = np.array(obj.noisestr[0]) / relative_x
    # xname = "$\sigma \mathrm{[fT]}$"
    # xname = "$ SNR   \mathrm{[dB]}$"
    xname = "ratio"

    # fig1, ax1 = plot_one_var(obj.noisestr[0], obj.avg_avgcc, yerr=obj.std_avgcc, xname=xname, yname="$\mathrm{CC}$",
    #                          xscale=xscale, label_name=labels[0], legend=True)
    # fig2, ax2 = plot_one_var(obj.noisestr[0], obj.avg_avgre, yerr=obj.std_avgre, xname=xname, yname="$\mathrm{RE}$",
    #                          xscale=xscale, label_name=labels[0], legend=True)
    fig3, ax3 = plot_one_var(ratios, obj.avg_avgdist, yerr=obj.std_avgdist, xname=xname,
                             yname="$d_{\mathrm{s,f}} \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=True)
    # fig4, ax4 = plot_one_var(obj.noisestr[0], obj.avg_snr, yerr=obj.std_snr, xname=xname, yname="$\mathrm{SNR}$",
    #                          xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        res1 = [idx for idx, val in enumerate(obj1.spontnoise[0]) if val == 0.0]
        res2 = [idx for idx, val in enumerate(obj1.noisestr[0]) if relative_x-(0.01) * 10 ** (-14.0) <= val < relative_x+(0.01) * 10 ** (-14.0)]

        res = [value for value in res1 if value in res2]
        ax3.hlines(obj1.avg_avgdist[res]*10**3, min(ratios), max(ratios), colors="red", linestyles='solid',
                   label=labels[1])
        ax3.legend()

        #
        #  add_one_var_fig(fig1, ax1, obj1.noisestr[0], obj1.avg_avgcc, yerr=obj1.std_avgcc, xscale=xscale, color="red",
        #                  label_name=labels[1], legend=True)
        #  add_one_var_fig(fig2, ax2, obj1.noisestr[0], obj1.avg_avgre, yerr=obj1.std_avgre, xscale=xscale, color="red",
        #                  label_name=labels[1], legend=True)
        # add_one_var_fig(fig3, ax3, obj1.noisestr[0], obj1.avg_avgdist, yerr=obj1.std_avgdist, xscale=xscale,
        #                 yscale=10 ** 3, color="red", label_name=labels[1], legend=True)
        # add_one_var_fig(fig4, ax4, obj1.noisestr[0], obj1.avg_snr, yerr=obj1.std_snr, xscale=xscale, color="red",
        #               label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        ax3.hlines(obj2.avg_avgdist[res]*10**3, min(ratios), max(ratios), colors="green", linestyles='solid',
                   label=labels[2])
        ax3.legend()
        # add_one_var_fig(fig1, ax1, obj2.noisestr[0], obj2.avg_avgcc, yerr=obj2.std_avgcc, xscale=xscale, color="blue",
        #                 label_name=labels[2], legend=True)
        # add_one_var_fig(fig2, ax2, obj2.noisestr[0], obj2.avg_avgre, yerr=obj2.std_avgre, xscale=xscale, color="blue",
        #                 label_name=labels[2], legend=True)
        # add_one_var_fig(fig3, ax3, obj2.noisestr[0], obj2.avg_avgdist, yerr=obj2.std_avgdist, xscale=xscale,
        #                 yscale=10 ** 3, color="blue", label_name=labels[2], legend=True)
        # add_one_var_fig(fig4, ax4, obj2.noisestr[0], obj2.avg_snr, yerr=obj2.std_snr, xscale=xscale, color="blue",
        #                 label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        ax3.hlines(obj3.avg_avgdist[res]*10**3, min(ratios), max(ratios), colors="blue", linestyles='solid',
                   label=labels[3])
        ax3.legend()
        # add_one_var_fig(fig1, ax1, obj3.noisestr[0], obj3.avg_avgcc, yerr=obj3.std_avgcc, xscale=xscale, color="green",
        #                 label_name=labels[3], legend=True, savefig=labels[0] + "_components_cc.png")
        # add_one_var_fig(fig2, ax2, obj3.noisestr[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=xscale, color="green",
        #                 label_name=labels[3], legend=True, savefig=labels[0] + "_components_re.png")
        # add_one_var_fig(fig3, ax3, obj3.noisestr[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=xscale,
        #                 yscale=10 ** 3,
        #                 color="green", label_name=labels[3], legend=True,
        #                 savefig=labels[0] + "_components_dist.png")
        # add_one_var_fig(fig4, ax4, obj3.noisestr[0], obj3.avg_snr, yerr=obj3.std_snr, xscale=xscale, color="green",
        #                 label_name=labels[3], legend=True, savefig=labels[0] + "_components_snr.png")

    plt.savefig(fname[:-3] + "png", dpi=300)
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


def test(fname, fname1=None, fname2=None, fname3=None, labels=["empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()

    return


def plot_contour_SNR(fname, fname1=None, fname2=None, fname3=None, labels=["empty", "empty", "empty", "empty"], parameter="dist"):
    import matplotlib.pyplot as plt
    import matplotlib.mlab as ml
    import matplotlib
    import numpy as np
    from scipy.interpolate import griddata

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 14}

    matplotlib.rc('font', **font)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    obj = read_obj(fname)
    obj.avg_snr = np.average(np.array(obj.avgsnr), axis=0)
    obj.avg_snrdb = np.average(np.array(obj.avgsnrdb), axis=0)
    z = obj.avg_snr

    z1 = np.zeros(np.shape(z))
    z2 = np.zeros(np.shape(z))
    z3 = np.zeros(np.shape(z))
    instances = 1

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.avg_snr = np.average(np.array(obj1.avgsnr), axis=0)
        obj1.avg_snrdb = np.average(np.array(obj1.avgsnrdb), axis=0)
        z1 = obj.avg_snr
        instances += 1

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.avg_snr = np.average(np.array(obj2.avgsnr), axis=0)
        obj2.avg_snrdb = np.average(np.array(obj2.avgsnrdb), axis=0)
        z2 = obj.avg_snr
        instances += 1

    if isinstance(fname2, str):
        obj3 = read_obj(fname3)
        obj3.avg_snr = np.average(np.array(obj3.avgsnr), axis=0)
        obj3.avg_snrdb = np.average(np.array(obj3.avgsnrdb), axis=0)
        z3 = obj.avg_snr
        instances += 1

    inf_zero=0

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

        # plt.contour(xi, yi, zi, levels=np.linspace(zmin,zmax,15), linewidths=0.5, colors='k')
        plt.pcolormesh(xi, yi, zi, cmap=plt.get_cmap('hot'), vmin=zmin, vmax=zmax)

        clb = plt.colorbar()
        plt.scatter(x, y, marker='o', c='black', s=10, zorder=10)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("$\sigma \mathrm{[fT]}$")
        plt.ylabel("$|| q_{\mathrm{spont}} ||  \mathrm{[nAm]}$")
        clb.ax.set_title(zlabel)
        plt.savefig(fname[:-3]+"-"+labels[i_inst]+"-"+parameter+".png")
        plt.show()

    return


def plot_contour(fname, fname1=None, fname2=None, fname3=None, labels=["empty", "empty", "empty", "empty"], parameter="dist"):
    import matplotlib.pyplot as plt
    import matplotlib.mlab as ml
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 14}

    matplotlib.rc('font', **font)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

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
        zlabel = "$d_{\mathrm{s,f}} \mathrm{[mm]}$"
        z = obj.avg_avgdist*10**3
        z1 = obj1.avg_avgdist*10**3
        z2 = obj2.avg_avgdist*10**3
        z3 = obj3.avg_avgdist*10**3
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

        # plt.contour(xi, yi, zi, levels=np.linspace(zmin,zmax,15), linewidths=0.5, colors='k')
        plt.pcolormesh(xi, yi, zi, cmap=plt.get_cmap('hot'), vmin=zmin, vmax=zmax)

        clb = plt.colorbar()
        plt.scatter(x, y, marker='o', c='black', s=10, zorder=10)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("$\sigma \mathrm{[fT]}$")
        plt.ylabel("$q_{\mathrm{spont}} \mathrm{[nAm]}$")
        clb.ax.set_title(zlabel)
        plt.savefig(fname[:-3]+"-"+labels[i_inst]+"-"+parameter+".png", dpi=300)
        plt.show()

    return


def visualize_SNR(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt

    obj = read_obj(fname)
    obj.avg_snr = np.average(np.array(obj.avgsnr), axis=0)
    obj.avg_snrdb = np.average(np.array(obj.avgsnrdb), axis=0)
    obj.std_snr = np.std(np.array(obj.avgsnr), axis=0)
    obj.std_snrdb = np.std(np.array(obj.avgsnrdb), axis=0)

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.avg_snr = np.average(np.array(obj1.avgsnr), axis=0)
        obj1.avg_snrdb = np.average(np.array(obj1.avgsnrdb), axis=0)
        obj1.std_snr = np.std(np.array(obj1.avgsnr), axis=0)
        obj1.std_snrdb = np.std(np.array(obj1.avgsnrdb), axis=0)

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.avg_snr = np.average(np.array(obj2.avgsnr), axis=0)
        obj2.avg_snrdb = np.average(np.array(obj2.avgsnrdb), axis=0)
        obj2.std_snr = np.std(np.array(obj2.avgsnr), axis=0)
        obj2.std_snrdb = np.std(np.array(obj2.avgsnrdb), axis=0)

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.avg_snr = np.average(np.array(obj3.avgsnr), axis=0)
        obj3.avg_snrdb = np.average(np.array(obj3.avgsnrdb), axis=0)
        obj3.std_snr = np.std(np.array(obj3.avgsnr), axis=0)
        obj3.std_snrdb = np.std(np.array(obj3.avgsnrdb), axis=0)

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.avg_snr = np.average(np.array(obj4.avgsnr), axis=0)
        obj4.avg_snrdb = np.average(np.array(obj4.avgsnrdb), axis=0)
        obj4.std_snr = np.std(np.array(obj4.avgsnr), axis=0)
        obj4.std_snrdb = np.std(np.array(obj4.avgsnrdb), axis=0)

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.avg_snr = np.average(np.array(obj5.avgsnr), axis=0)
        obj5.avg_snrdb = np.average(np.array(obj5.avgsnrdb), axis=0)
        obj5.std_snr = np.std(np.array(obj5.avgsnr), axis=0)
        obj5.std_snrdb = np.std(np.array(obj5.avgsnrdb), axis=0)

    fig4, ax4 = plot_one_var(obj.noisestr[0], obj.avg_snrdb, yerr=obj.std_snrdb, xname="$\sigma \mathrm{[fT]}$",
                             yname="$\mathrm{SNR [dB]}$", xscale=10 ** 15, label_name=labels[0] , legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig4, ax4, obj1.noisestr[0], obj1.avg_snrdb, yerr=obj1.std_snrdb, xscale=10 ** 15, color="red",
                        label_name=labels[1] , legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig4, ax4, obj2.noisestr[0], obj2.avg_snrdb, yerr=obj2.std_snrdb, xscale=10 ** 15, color="blue",
                        label_name=labels[2] , legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig4, ax4, obj3.noisestr[0], obj3.avg_snrdb, yerr=obj3.std_snrdb, xscale=10 ** 15, color="green",
                        label_name=labels[3] , legend=True)

    if isinstance(fname4, str):
        add_one_var_fig(fig4, ax4, obj4.noisestr[0], obj4.avg_snrdb, yerr=obj4.std_snrdb, xscale=10 ** 15, color="cyan",
                        label_name=labels[4] , legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig4, ax4, obj5.noisestr[0], obj5.avg_snrdb, yerr=obj5.std_snrdb, xscale=10 ** 15, color="pink",
                        label_name=labels[5] , legend=True)

    fig4.savefig(fname[:-4] + "-snrdb.png", dpi=300)
    plt.show()

    return


def visualize(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt

    if isinstance(fname, str):
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

    fig1, ax1 = plot_one_var(obj.noisestr[0], obj.avg_avgcc, yerr=obj.std_avgcc, xname="$\sigma \mathrm{[fT]}$",
                             yname="$\mathrm{CC}$", xscale=10 ** 15, label_name=labels[0] , legend=True)
    fig2, ax2 = plot_one_var(obj.noisestr[0], obj.avg_avgre, yerr=obj.std_avgre, xname="$\sigma \mathrm{[fT]}$",
                             yname="$\mathrm{RE}$", xscale=10 ** 15, label_name=labels[0] , legend=True)
    fig3, ax3 = plot_one_var(obj.noisestr[0], obj.avg_avgdist, yerr=obj.std_avgdist, xname="$\sigma \mathrm{[fT]}$",
                             yname="$d{\mathrm{s,f}} \mathrm{[mm]}$", xscale=10 ** 15, yscale=10 ** 3,
                             label_name=labels[0], legend=True)
    fig4, ax4 = plot_one_var(obj.noisestr[0], obj.avg_snr, yerr=obj.std_snr, xname="$\sigma \mathrm{[fT]}$",
                             yname="$\mathrm{SNR}$", xscale=10 ** 15, label_name=labels[0] , legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig1, ax1, obj1.noisestr[0], obj1.avg_avgcc, yerr=obj1.std_avgcc, xscale=10 ** 15, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig2, ax2, obj1.noisestr[0], obj1.avg_avgre, yerr=obj1.std_avgre, xscale=10 ** 15, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig3, ax3, obj1.noisestr[0], obj1.avg_avgdist, yerr=obj1.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="red", label_name=labels[1] , legend=True)
        add_one_var_fig(fig4, ax4, obj1.noisestr[0], obj1.avg_snr, yerr=obj1.std_snr, xscale=10 ** 15, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig1, ax1, obj2.noisestr[0], obj2.avg_avgcc, yerr=obj2.std_avgcc, xscale=10 ** 15, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig2, ax2, obj2.noisestr[0], obj2.avg_avgre, yerr=obj2.std_avgre, xscale=10 ** 15, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig3, ax3, obj2.noisestr[0], obj2.avg_avgdist, yerr=obj2.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="blue", label_name=labels[2] , legend=True)
        add_one_var_fig(fig4, ax4, obj2.noisestr[0], obj2.avg_snr, yerr=obj2.std_snr, xscale=10 ** 15, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig1, ax1, obj3.noisestr[0], obj3.avg_avgcc, yerr=obj3.std_avgcc, xscale=10 ** 15, color="green",
                        label_name=labels[3], legend=True)
        add_one_var_fig(fig2, ax2, obj3.noisestr[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=10 ** 15, color="green",
                        label_name=labels[3], legend=True)
        add_one_var_fig(fig3, ax3, obj3.noisestr[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="green", label_name=labels[3], legend=True)
        add_one_var_fig(fig4, ax4, obj3.noisestr[0], obj3.avg_snr, yerr=obj3.std_snr, xscale=10 ** 15, color="green",
                        label_name=labels[3], legend=True)

    if isinstance(fname4, str):
        add_one_var_fig(fig1, ax1, obj4.noisestr[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=10 ** 15, color="cyan",
                        label_name=labels[4], legend=True)
        add_one_var_fig(fig2, ax2, obj4.noisestr[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=10 ** 15, color="cyan",
                        label_name=labels[4], legend=True)
        add_one_var_fig(fig3, ax3, obj4.noisestr[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="cyan", label_name=labels[4] , legend=True)
        add_one_var_fig(fig4, ax4, obj4.noisestr[0], obj4.avg_snr, yerr=obj4.std_snr, xscale=10 ** 15, color="cyan",
                        label_name=labels[4], legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig1, ax1, obj5.noisestr[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=10 ** 15, color="pink",
                        label_name=labels[5], legend=True)
        add_one_var_fig(fig2, ax2, obj5.noisestr[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=10 ** 15, color="pink",
                        label_name=labels[5], legend=True)
        add_one_var_fig(fig3, ax3, obj5.noisestr[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=10 ** 15,
                        yscale=10 ** 3, color="pink", label_name=labels[5] , legend=True)
        add_one_var_fig(fig4, ax4, obj5.noisestr[0], obj5.avg_snr, yerr=obj5.std_snr, xscale=10 ** 15, color="pink",
                        label_name=labels[5], legend=True)

    fig1.savefig(fname[:-4] + "-cc.png", dpi=300)
    fig2.savefig(fname[:-4] + "-re.png", dpi=300)
    fig3.savefig(fname[:-4] + "-dist.png", dpi=300)
    fig4.savefig(fname[:-4] + "-snr.png", dpi=300)
    plt.show()

    return


def visualize2_SNR(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.avg_snr = np.average(np.array(obj.avgsnr), axis=0)
    obj.avg_snrdb = np.average(np.array(obj.avgsnrdb), axis=0)
    obj.std_snr = np.std(np.array(obj.avgsnr), axis=0)
    obj.std_snrdb = np.std(np.array(obj.avgsnrdb), axis=0)

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.avg_snr = np.average(np.array(obj1.avgsnr), axis=0)
        obj1.avg_snrdb = np.average(np.array(obj1.avgsnrdb), axis=0)
        obj1.std_snr = np.std(np.array(obj1.avgsnr), axis=0)
        obj1.std_snrdb = np.std(np.array(obj1.avgsnrdb), axis=0)

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.avg_snr = np.average(np.array(obj2.avgsnr), axis=0)
        obj2.avg_snrdb = np.average(np.array(obj2.avgsnrdb), axis=0)
        obj2.std_snr = np.std(np.array(obj2.avgsnr), axis=0)
        obj2.std_snrdb = np.std(np.array(obj2.avgsnrdb), axis=0)

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.avg_snr = np.average(np.array(obj3.avgsnr), axis=0)
        obj3.avg_snrdb = np.average(np.array(obj3.avgsnrdb), axis=0)
        obj3.std_snr = np.std(np.array(obj3.avgsnr), axis=0)
        obj3.std_snrdb = np.std(np.array(obj3.avgsnrdb), axis=0)

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.avg_snr = np.average(np.array(obj4.avgsnr), axis=0)
        obj4.avg_snrdb = np.average(np.array(obj4.avgsnrdb), axis=0)
        obj4.std_snr = np.std(np.array(obj4.avgsnr), axis=0)
        obj4.std_snrdb = np.std(np.array(obj4.avgsnrdb), axis=0)

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.avg_snr = np.average(np.array(obj5.avgsnr), axis=0)
        obj5.avg_snrdb = np.average(np.array(obj5.avgsnrdb), axis=0)
        obj5.std_snr = np.std(np.array(obj5.avgsnr), axis=0)
        obj5.std_snrdb = np.std(np.array(obj5.avgsnrdb), axis=0)


    # xscale = 10 ** 15
    xscale = 10 ** 9
    # xname = "$\sigma \mathrm{[fT]}$"
    xname = "$\| q_{\mathrm{spont}} \|  \mathrm{[nAm]}$"

    fig4, ax4 = plot_one_var(obj.spontnoise[0], obj.avg_snrdb, yerr=obj.std_snrdb, xname=xname, yname="$\mathrm{SNR [dB]}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig4, ax4, obj1.spontnoise[0], obj1.avg_snrdb, yerr=obj1.std_snrdb, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig4, ax4, obj2.spontnoise[0], obj2.avg_snrdb, yerr=obj2.std_snrdb, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig4, ax4, obj3.spontnoise[0], obj3.avg_snrdb, yerr=obj3.std_snrdb, xscale=xscale, color="green",
                        label_name=labels[3], legend=True)

    if isinstance(fname4, str):
        add_one_var_fig(fig4, ax4, obj4.spontnoise[0], obj4.avg_snrdb, yerr=obj4.std_snrdb, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig4, ax4, obj5.spontnoise[0], obj5.avg_snrdb, yerr=obj5.std_snrdb, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)

    fig4.savefig(fname[:-4] + "-snrdb.png", dpi=300)
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
    xname = "$q_{\mathrm{spont}}   \mathrm{[nAm]}$"

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
                        label_name=labels[3], legend=True)
        add_one_var_fig(fig2, ax2, obj3.spontnoise[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=xscale, color="green",
                        label_name=labels[3], legend=True)
        add_one_var_fig(fig3, ax3, obj3.spontnoise[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=xscale,
                        yscale=10 ** 3,
                        color="green", label_name=labels[3], legend=True)
        add_one_var_fig(fig4, ax4, obj3.spontnoise[0], obj3.avg_snr, yerr=obj3.std_snr, xscale=xscale, color="green",
                        label_name=labels[3], legend=True)

    if isinstance(fname4, str):
        add_one_var_fig(fig1, ax1, obj4.spontnoise[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)
        add_one_var_fig(fig2, ax2, obj4.spontnoise[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)
        add_one_var_fig(fig3, ax3, obj4.spontnoise[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="cyan", label_name=labels[4], legend=True,)
        add_one_var_fig(fig4, ax4, obj4.spontnoise[0], obj4.avg_snr, yerr=obj4.std_snr, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig1, ax1, obj5.spontnoise[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)
        add_one_var_fig(fig2, ax2, obj5.spontnoise[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)
        add_one_var_fig(fig3, ax3, obj5.spontnoise[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="pink", label_name=labels[5], legend=True)
        add_one_var_fig(fig4, ax4, obj5.spontnoise[0], obj5.avg_snr, yerr=obj5.std_snr, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)

    fig1.savefig(fname[:-4] + "-cc.png", dpi=300)
    fig2.savefig(fname[:-4] + "-re.png", dpi=300)
    fig3.savefig(fname[:-4] + "-dist.png", dpi=300)
    fig4.savefig(fname[:-4] + "-snr.png", dpi=300)
    plt.show()

    return


def visualize3(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"], addtoname=""):
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
    xname = "$ \mathrm{SNR}   \mathrm{[dB]}$"

    fig1, ax1 = plot_one_var(obj.noisestr[0], obj.avg_avgcc, yerr=obj.std_avgcc, xname=xname, yname="$\mathrm{CC}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig2, ax2 = plot_one_var(obj.noisestr[0], obj.avg_avgre, yerr=obj.std_avgre, xname=xname, yname="$\mathrm{RE}$",
                             xscale=xscale, label_name=labels[0], legend=True)
    fig3, ax3 = plot_one_var(obj.noisestr[0], obj.avg_avgdist, yerr=obj.std_avgdist, xname=xname,
                             yname="$d(p1, p2) \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=True)
    fig4, ax4 = plot_one_var(obj.noisestr[0], obj.avg_snrdb, yerr=obj.std_snrdb, xname=xname, yname="$\mathrm{SNR}   \mathrm{[dB]}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig1, ax1, obj1.noisestr[0], obj1.avg_avgcc, yerr=obj1.std_avgcc, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig2, ax2, obj1.noisestr[0], obj1.avg_avgre, yerr=obj1.std_avgre, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)
        add_one_var_fig(fig3, ax3, obj1.noisestr[0], obj1.avg_avgdist, yerr=obj1.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="red", label_name=labels[1], legend=True)
        add_one_var_fig(fig4, ax4, obj1.noisestr[0], obj1.avg_snrdb, yerr=obj1.std_snrdb, xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig1, ax1, obj2.noisestr[0], obj2.avg_avgcc, yerr=obj2.std_avgcc, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig2, ax2, obj2.noisestr[0], obj2.avg_avgre, yerr=obj2.std_avgre, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)
        add_one_var_fig(fig3, ax3, obj2.noisestr[0], obj2.avg_avgdist, yerr=obj2.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="blue", label_name=labels[2], legend=True)
        add_one_var_fig(fig4, ax4, obj2.noisestr[0], obj2.avg_snrdb, yerr=obj2.std_snrdb, xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig1, ax1, obj3.noisestr[0], obj3.avg_avgcc, yerr=obj3.std_avgcc, xscale=xscale, color="green",
                        label_name=labels[3], legend=True)
        add_one_var_fig(fig2, ax2, obj3.noisestr[0], obj3.avg_avgre, yerr=obj3.std_avgre, xscale=xscale, color="green",
                        label_name=labels[3], legend=True)
        add_one_var_fig(fig3, ax3, obj3.noisestr[0], obj3.avg_avgdist, yerr=obj3.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="green", label_name=labels[3], legend=True)
        add_one_var_fig(fig4, ax4, obj3.noisestr[0], obj3.avg_snrdb, yerr=obj3.std_snrdb, xscale=xscale, color="green",
                        label_name=labels[3], legend=True)

    if isinstance(fname4, str):
        add_one_var_fig(fig1, ax1, obj4.noisestr[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)
        add_one_var_fig(fig2, ax2, obj4.noisestr[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)
        add_one_var_fig(fig3, ax3, obj4.noisestr[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="cyan", label_name=labels[4], legend=True)
        add_one_var_fig(fig4, ax4, obj4.noisestr[0], obj4.avg_snrdb, yerr=obj4.std_snrdb, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig1, ax1, obj5.noisestr[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)
        add_one_var_fig(fig2, ax2, obj5.noisestr[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)
        add_one_var_fig(fig3, ax3, obj5.noisestr[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=xscale,
                        yscale=10 ** 3, color="pink", label_name=labels[5], legend=True)
        add_one_var_fig(fig4, ax4, obj5.noisestr[0], obj5.avg_snrdb, yerr=obj5.std_snrdb, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)

    ax1.invert_xaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()
    ax4.invert_xaxis()

    fig1.savefig(fname[:-4] + "-cc" + addtoname + ".png", dpi=300)
    fig2.savefig(fname[:-4] + "-re" + addtoname + ".png", dpi=300)
    fig3.savefig(fname[:-4] + "-dist" + addtoname + ".png", dpi=300)
    fig4.savefig(fname[:-4] + "-snrdb" + addtoname + ".png", dpi=300)

    plt.show()

    return


def visualize4(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None, fname6=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty", "empty"], addtoname=""):
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

    if isinstance(fname6, str):
        obj6 = read_obj(fname6)
        obj6.calc_avgs()
        obj6.calc_stds()

    xscale = 10 ** 15
    # xscale = 1

    xname = "$\sigma \mathrm{[fT]}$"

    # np.argwhere(np.array(obj.spontnoise[0]) == 0.0).tolist()
    res = [idx for idx, val in enumerate(obj.spontnoise[0]) if val == 0.0]

    fig1, ax1 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_avgcc)[res], yerr=np.array(obj.std_avgcc)[res], xname=xname, yname="$\mathrm{CC}$",
                             xscale=xscale, label_name=labels[0], legend=False, marker="X")
    fig2, ax2 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_avgre)[res], yerr=np.array(obj.std_avgre)[res], xname=xname, yname="$\mathrm{RE}$",
                             xscale=xscale, label_name=labels[0], legend=False, marker="X")
    fig3, ax3 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_avgdist)[res], yerr=np.array(obj.std_avgdist)[res], xname=xname,
                             yname="$d_{\mathrm{s,f}} \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=False, marker="X")
    fig4, ax4 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_snrdb)[res], yerr=np.array(obj.std_snrdb)[res], xname=xname, yname="$\mathrm{SNR [dB]}$",
                             xscale=xscale, label_name=labels[0], legend=False, marker="X")

    if isinstance(fname1, str):
        ccolor = "red"
        cmarker = "s"

        res = [idx for idx, val in enumerate(obj1.spontnoise[0]) if val == 0.0]
        add_one_var_fig(fig1, ax1, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_avgcc)[res], yerr=np.array(obj1.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[1], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_avgre)[res], yerr=np.array(obj1.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[1], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_avgdist)[res], yerr=np.array(obj1.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[1], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_snrdb)[res], yerr=np.array(obj1.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[1], legend=False, marker=cmarker)

    if isinstance(fname2, str):
        ccolor = "green"
        cmarker = "o"

        res = [idx for idx, val in enumerate(obj2.spontnoise[0]) if val == 0.0]
        add_one_var_fig(fig1, ax1, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_avgcc)[res], yerr=np.array(obj2.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[2], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_avgre)[res], yerr=np.array(obj2.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[2], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_avgdist)[res], yerr=np.array(obj2.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[2], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_snrdb)[res], yerr=np.array(obj2.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[2], legend=False, marker=cmarker)

    if isinstance(fname3, str):
        ccolor = "blue"
        cmarker = "^"

        res = [idx for idx, val in enumerate(obj3.spontnoise[0]) if val == 0.0]
        add_one_var_fig(fig1, ax1, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_avgcc)[res], yerr=np.array(obj3.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_avgre)[res], yerr=np.array(obj3.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_avgdist)[res], yerr=np.array(obj3.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_snrdb)[res], yerr=np.array(obj3.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)

    if isinstance(fname4, str):
        ccolor = "orange"
        cmarker = "P"

        res = [idx for idx, val in enumerate(obj4.spontnoise[0]) if val == 0.0]
        add_one_var_fig(fig1, ax1, np.array(obj4.noisestr[0])[res], np.array(obj4.avg_avgcc)[res], yerr=np.array(obj4.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[4], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj4.noisestr[0])[res], np.array(obj4.avg_avgre)[res], yerr=np.array(obj4.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[4], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj4.noisestr[0])[res], np.array(obj4.avg_avgdist)[res], yerr=np.array(obj4.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[4], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj4.noisestr[0])[res], np.array(obj4.avg_snrdb)[res], yerr=np.array(obj4.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[4], legend=False, marker=cmarker)

    if isinstance(fname5, str):
        ccolor = "purple"
        cmarker = "D"

        res = [idx for idx, val in enumerate(obj5.spontnoise[0]) if val == 0.0]
        add_one_var_fig(fig1, ax1, np.array(obj5.noisestr[0])[res], np.array(obj5.avg_avgcc)[res], yerr=np.array(obj5.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[5], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj5.noisestr[0])[res], np.array(obj5.avg_avgre)[res], yerr=np.array(obj5.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj5.noisestr[0])[res], np.array(obj5.avg_avgdist)[res], yerr=np.array(obj5.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[5], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj5.noisestr[0])[res], np.array(obj5.avg_snrdb)[res], yerr=np.array(obj5.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[5], legend=False, marker=cmarker)

    if isinstance(fname6, str):
        ccolor = "brown"
        cmarker = "*"

        res = [idx for idx, val in enumerate(obj6.spontnoise[0]) if val == 0.0]
        add_one_var_fig(fig1, ax1, np.array(obj6.noisestr[0])[res], np.array(obj6.avg_avgcc)[res], yerr=np.array(obj6.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj6.noisestr[0])[res], np.array(obj6.avg_avgre)[res], yerr=np.array(obj6.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj6.noisestr[0])[res], np.array(obj6.avg_avgdist)[res], yerr=np.array(obj6.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj6.noisestr[0])[res], np.array(obj6.avg_snrdb)[res], yerr=np.array(obj6.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[6], legend=False, marker=cmarker)

    fig1.savefig(fname[:-4] + "-sigma-cc" + addtoname +".png", dpi=300)
    fig2.savefig(fname[:-4] + "-sigma-re" + addtoname +".png", dpi=300)
    fig3.savefig(fname[:-4] + "-sigma-dist" + addtoname +".png", dpi=300)
    fig4.savefig(fname[:-4] + "-sigma-snrdb" + addtoname +".png", dpi=300)

    figlegend = plt.figure(figsize=(12, 1))
    axl = figlegend.add_subplot(111)
    plt.figlegend(*ax1.get_legend_handles_labels(), loc='center', ncol=3, prop={'size': 15})
    axl.set_axis_off()
    figlegend.savefig(fname[:-4] + "simulation_legend.png", dpi=300)
    plt.show()

    return


def visualize4_SNR(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.avg_snr = np.average(np.array(obj.avgsnr), axis=0)
    obj.avg_snrdb = np.average(np.array(obj.avgsnrdb), axis=0)
    obj.std_snr = np.std(np.array(obj.avgsnr), axis=0)
    obj.std_snrdb = np.std(np.array(obj.avgsnrdb), axis=0)

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.avg_snr = np.average(np.array(obj1.avgsnr), axis=0)
        obj1.avg_snrdb = np.average(np.array(obj1.avgsnrdb), axis=0)
        obj1.std_snr = np.std(np.array(obj1.avgsnr), axis=0)
        obj1.std_snrdb = np.std(np.array(obj1.avgsnrdb), axis=0)

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.avg_snr = np.average(np.array(obj2.avgsnr), axis=0)
        obj2.avg_snrdb = np.average(np.array(obj2.avgsnrdb), axis=0)
        obj2.std_snr = np.std(np.array(obj2.avgsnr), axis=0)
        obj2.std_snrdb = np.std(np.array(obj2.avgsnrdb), axis=0)

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.avg_snr = np.average(np.array(obj3.avgsnr), axis=0)
        obj3.avg_snrdb = np.average(np.array(obj3.avgsnrdb), axis=0)
        obj3.std_snr = np.std(np.array(obj3.avgsnr), axis=0)
        obj3.std_snrdb = np.std(np.array(obj3.avgsnrdb), axis=0)

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.avg_snr = np.average(np.array(obj4.avgsnr), axis=0)
        obj4.avg_snrdb = np.average(np.array(obj4.avgsnrdb), axis=0)
        obj4.std_snr = np.std(np.array(obj4.avgsnr), axis=0)
        obj4.std_snrdb = np.std(np.array(obj4.avgsnrdb), axis=0)

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.avg_snr = np.average(np.array(obj5.avgsnr), axis=0)
        obj5.avg_snrdb = np.average(np.array(obj5.avgsnrdb), axis=0)
        obj5.std_snr = np.std(np.array(obj5.avgsnr), axis=0)
        obj5.std_snrdb = np.std(np.array(obj5.avgsnrdb), axis=0)

    xscale = 10 ** 15
    # xscale = 1
    xname = "$\sigma \mathrm{[fT]}$"
    # xname = "$\| SNR \|  \mathrm{[dB]}$"

    # np.argwhere(np.array(obj.spontnoise[0]) == 0.0).tolist()
    res = [idx for idx, val in enumerate(obj.spontnoise[0]) if val == 0.0]

    fig4, ax4 = plot_one_var(np.array(obj.noisestr[0])[res], np.array(obj.avg_snrdb)[res], yerr=np.array(obj.std_snrdb)[res], xname=xname, yname="$\mathrm{SNR [dB]}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig4, ax4, np.array(obj1.noisestr[0])[res], np.array(obj1.avg_snrdb)[res], yerr=np.array(obj1.std_snrdb)[res], xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig4, ax4, np.array(obj2.noisestr[0])[res], np.array(obj2.avg_snrdb)[res], yerr=np.array(obj2.std_snrdb)[res], xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig4, ax4, np.array(obj3.noisestr[0])[res], np.array(obj3.avg_snrdb)[res], yerr=np.array(obj3.std_snrdb)[res], xscale=xscale, color="green",
                        label_name=labels[3], legend=True)

    if isinstance(fname4, str):
        add_one_var_fig(fig4, ax4, obj4.noisestr[0], obj4.avg_snrdb, yerr=obj4.std_snrdb, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig4, ax4, obj5.noisestr[0], obj5.avg_snrdb, yerr=obj5.std_snrdb, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)

    fig4.savefig(fname[:-4] + "-sigma-snrdb-extra.png", dpi=300)

    plt.show()

    return


def visualize5(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None, fname6=None, fname7=None,
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

    if isinstance(fname6, str):
        obj6 = read_obj(fname6)
        obj6.calc_avgs()
        obj6.calc_stds()

    if isinstance(fname7, str):
        obj7 = read_obj(fname7)
        obj7.calc_avgs()
        obj7.calc_stds()

    xscale = 10 ** 9
    # xscale = 1
    # xname = "$\sigma \mathrm{[fT]}$"
    # xname = "$\| SNR \|  \mathrm{[dB]}$"
    xname = "$q_{\mathrm{spont}}  \mathrm{[nAm]}$"

    # np.argwhere(np.array(obj.spontnoise[0]) == 0.0).tolist()
    res = [idx for idx, val in enumerate(obj.noisestr[0]) if val == 0.0]

    fig1, ax1 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_avgcc)[res], yerr=np.array(obj.std_avgcc)[res], xname=xname, yname="$\mathrm{CC}$",
                             xscale=xscale, label_name=labels[0], legend=False, marker="X")
    fig2, ax2 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_avgre)[res], yerr=np.array(obj.std_avgre)[res], xname=xname, yname="$\mathrm{RE}$",
                             xscale=xscale, label_name=labels[0], legend=False, marker="X")
    fig3, ax3 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_avgdist)[res], yerr=np.array(obj.std_avgdist)[res], xname=xname,
                             yname="$d_{\mathrm{s,f}} \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
                             label_name=labels[0], legend=False, marker="X")
    fig4, ax4 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_snrdb)[res], yerr=np.array(obj.std_snrdb)[res], xname=xname, yname="$\mathrm{SNR [dB]}$",
                             xscale=xscale, label_name=labels[0], legend=False, marker="X")

    if isinstance(fname1, str):
        res = [idx for idx, val in enumerate(obj1.noisestr[0]) if val == 0.0]

        ccolor = "red"
        cmarker = "s"

        add_one_var_fig(fig1, ax1, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_avgcc)[res], yerr=np.array(obj1.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[1], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_avgre)[res], yerr=np.array(obj1.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[1], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_avgdist)[res], yerr=np.array(obj1.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[1], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_snrdb)[res], yerr=np.array(obj1.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[1], legend=False, marker=cmarker)

    if isinstance(fname2, str):
        res = [idx for idx, val in enumerate(obj2.noisestr[0]) if val == 0.0]

        ccolor = "green"
        cmarker = "o"

        add_one_var_fig(fig1, ax1, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_avgcc)[res], yerr=np.array(obj2.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[2], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_avgre)[res], yerr=np.array(obj2.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[2], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_avgdist)[res], yerr=np.array(obj2.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[2], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_snrdb)[res], yerr=np.array(obj2.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[2], legend=False, marker=cmarker)

    if isinstance(fname3, str):
        res = [idx for idx, val in enumerate(obj3.noisestr[0]) if val == 0.0]

        ccolor = "blue"
        cmarker = "^"

        add_one_var_fig(fig1, ax1, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_avgcc)[res], yerr=np.array(obj3.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_avgre)[res], yerr=np.array(obj3.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_avgdist)[res], yerr=np.array(obj3.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_snrdb)[res], yerr=np.array(obj3.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)

    if isinstance(fname4, str):
        res = [idx for idx, val in enumerate(obj4.noisestr[0]) if val == 0.0]

        ccolor = "orange"
        cmarker = "P"

        add_one_var_fig(fig1, ax1, np.array(obj4.spontnoise[0])[res], np.array(obj4.avg_avgcc)[res], yerr=np.array(obj4.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[4], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj4.spontnoise[0])[res], np.array(obj4.avg_avgre)[res], yerr=np.array(obj4.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[4], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj4.spontnoise[0])[res], np.array(obj4.avg_avgdist)[res], yerr=np.array(obj4.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[4], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj4.spontnoise[0])[res], np.array(obj4.avg_snrdb)[res], yerr=np.array(obj4.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[4], legend=False, marker=cmarker)

    if isinstance(fname5, str):
        res = [idx for idx, val in enumerate(obj5.noisestr[0]) if val == 0.0]

        ccolor = "purple"
        cmarker = "D"

        add_one_var_fig(fig1, ax1, np.array(obj5.spontnoise[0])[res], np.array(obj5.avg_avgcc)[res], yerr=np.array(obj5.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[5], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj5.spontnoise[0])[res], np.array(obj5.avg_avgre)[res], yerr=np.array(obj5.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[5], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj5.spontnoise[0])[res], np.array(obj5.avg_avgdist)[res], yerr=np.array(obj5.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[5], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj5.spontnoise[0])[res], np.array(obj5.avg_snrdb)[res], yerr=np.array(obj5.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[5], legend=False, marker=cmarker)

    if isinstance(fname6, str):
        res = [idx for idx, val in enumerate(obj6.noisestr[0]) if val == 0.0]

        ccolor = "brown"
        cmarker = "*"

        add_one_var_fig(fig1, ax1, np.array(obj6.spontnoise[0])[res], np.array(obj6.avg_avgcc)[res], yerr=np.array(obj6.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj6.spontnoise[0])[res], np.array(obj6.avg_avgre)[res], yerr=np.array(obj6.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj6.spontnoise[0])[res], np.array(obj6.avg_avgdist)[res], yerr=np.array(obj6.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj6.spontnoise[0])[res], np.array(obj6.avg_snrdb)[res], yerr=np.array(obj6.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[6], legend=False, marker=cmarker)

    if isinstance(fname7, str):
        res = [idx for idx, val in enumerate(obj7.noisestr[0]) if val == 0.0]
        ccolor = "gray"
        cmarker = "H"

        add_one_var_fig(fig1, ax1, np.array(obj7.spontnoise[0])[res], np.array(obj7.avg_avgcc)[res], yerr=np.array(obj7.std_avgcc)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig2, ax2, np.array(obj7.spontnoise[0])[res], np.array(obj7.avg_avgre)[res], yerr=np.array(obj7.std_avgre)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)
        add_one_var_fig(fig3, ax3, np.array(obj7.spontnoise[0])[res], np.array(obj7.avg_avgdist)[res], yerr=np.array(obj7.std_avgdist)[res], xscale=xscale,
                        yscale=10 ** 3, color=ccolor, label_name=labels[6], legend=False, marker=cmarker)
        add_one_var_fig(fig4, ax4, np.array(obj7.spontnoise[0])[res], np.array(obj7.avg_snrdb)[res], yerr=np.array(obj7.std_snrdb)[res], xscale=xscale, color=ccolor,
                        label_name=labels[3], legend=False, marker=cmarker)

    figlegend = plt.figure(figsize=(12, 1))
    axl = figlegend.add_subplot(111)
    plt.figlegend(*ax1.get_legend_handles_labels(), loc='center', ncol=3, prop={'size': 15})
    axl.set_axis_off()

    fig1.savefig(fname[:-4] + "-noisestr-cc-combined.png", dpi=300)
    fig2.savefig(fname[:-4] + "-noisestr-re-combined.png", dpi=300)
    fig3.savefig(fname[:-4] + "-noisestr-dist-combined.png", dpi=300)
    fig4.savefig(fname[:-4] + "-noisestr-snrdb-combined.png", dpi=300)

    figlegend.savefig(fname[:-4] + "simulation_legend_noisestr.png", dpi=300)

    plt.show()

    return


def visualize5_SNR(fname, fname1=None, fname2=None, fname3=None, fname4=None, fname5=None,
              labels=["empty", "empty", "empty", "empty", "empty", "empty"]):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.avg_snr = np.average(np.array(obj.avgsnr), axis=0)
    obj.avg_snrdb = np.average(np.array(obj.avgsnrdb), axis=0)
    obj.std_snr = np.std(np.array(obj.avgsnr), axis=0)
    obj.std_snrdb = np.std(np.array(obj.avgsnrdb), axis=0)

    if isinstance(fname1, str):
        obj1 = read_obj(fname1)
        obj1.avg_snr = np.average(np.array(obj1.avgsnr), axis=0)
        obj1.avg_snrdb = np.average(np.array(obj1.avgsnrdb), axis=0)
        obj1.std_snr = np.std(np.array(obj1.avgsnr), axis=0)
        obj1.std_snrdb = np.std(np.array(obj1.avgsnrdb), axis=0)

    if isinstance(fname2, str):
        obj2 = read_obj(fname2)
        obj2.avg_snr = np.average(np.array(obj2.avgsnr), axis=0)
        obj2.avg_snrdb = np.average(np.array(obj2.avgsnrdb), axis=0)
        obj2.std_snr = np.std(np.array(obj2.avgsnr), axis=0)
        obj2.std_snrdb = np.std(np.array(obj2.avgsnrdb), axis=0)

    if isinstance(fname3, str):
        obj3 = read_obj(fname3)
        obj3.avg_snr = np.average(np.array(obj3.avgsnr), axis=0)
        obj3.avg_snrdb = np.average(np.array(obj3.avgsnrdb), axis=0)
        obj3.std_snr = np.std(np.array(obj3.avgsnr), axis=0)
        obj3.std_snrdb = np.std(np.array(obj3.avgsnrdb), axis=0)

    if isinstance(fname4, str):
        obj4 = read_obj(fname4)
        obj4.avg_snr = np.average(np.array(obj4.avgsnr), axis=0)
        obj4.avg_snrdb = np.average(np.array(obj4.avgsnrdb), axis=0)
        obj4.std_snr = np.std(np.array(obj4.avgsnr), axis=0)
        obj4.std_snrdb = np.std(np.array(obj4.avgsnrdb), axis=0)

    if isinstance(fname5, str):
        obj5 = read_obj(fname5)
        obj5.avg_snr = np.average(np.array(obj5.avgsnr), axis=0)
        obj5.avg_snrdb = np.average(np.array(obj5.avgsnrdb), axis=0)
        obj5.std_snr = np.std(np.array(obj5.avgsnr), axis=0)
        obj5.std_snrdb = np.std(np.array(obj5.avgsnrdb), axis=0)

    xscale = 10 ** 9
    # xscale = 1
    # xname = "$\sigma \mathrm{[fT]}$"
    # xname = "$\| SNR \|  \mathrm{[dB]}$"
    xname = "$\| q_{\mathrm{spont}} \|   \mathrm{[nAm]}$"

    # np.argwhere(np.array(obj.spontnoise[0]) == 0.0).tolist()
    res = [idx for idx, val in enumerate(obj.noisestr[0]) if val == 0.0]

    fig4, ax4 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_snrdb)[res], yerr=np.array(obj.std_snrdb)[res], xname=xname, yname="$\mathrm{SNR [dB]}$",
                             xscale=xscale, label_name=labels[0], legend=True)

    if isinstance(fname1, str):
        add_one_var_fig(fig4, ax4, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_snrdb)[res], yerr=np.array(obj1.std_snrdb)[res], xscale=xscale, color="red",
                        label_name=labels[1], legend=True)

    if isinstance(fname2, str):
        add_one_var_fig(fig4, ax4, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_snrdb)[res], yerr=np.array(obj2.std_snrdb)[res], xscale=xscale, color="blue",
                        label_name=labels[2], legend=True)

    if isinstance(fname3, str):
        add_one_var_fig(fig4, ax4, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_snrdb)[res], yerr=np.array(obj3.std_snrdb)[res], xscale=xscale, color="green",
                        label_name=labels[3], legend=True)

    if isinstance(fname4, str):
        add_one_var_fig(fig4, ax4, obj4.spontnoise[0], obj4.avg_snrdb, yerr=obj4.std_snrdb, xscale=xscale, color="cyan",
                        label_name=labels[4], legend=True)

    if isinstance(fname5, str):
        add_one_var_fig(fig4, ax4, obj5.spontnoise[0], obj5.avg_snrdb, yerr=obj5.std_snrdb, xscale=xscale, color="pink",
                        label_name=labels[5], legend=True)

    fig4.savefig(fname[:-4] + "-noisestr-snrdb-extra.png", dpi=300)

    plt.show()

    return


def visualize_uniform(fname, y_attr, x_attr, yerr_attr=None, fig=None, ax=None, xname="", yname="", xscale=1.0,
                      yscale=1.0, label="", color="black", marker="o"):
    import matplotlib.pyplot as plt
    obj = read_obj(fname)
    obj.calc_avgs()
    obj.calc_stds()

    yval = getattr(obj, y_attr)
    xval = getattr(obj, x_attr)[0]

    if yerr_attr is not None:
        yerrval = getattr(obj, yerr_attr)
    else:
        yerrval = []

    if fig is None and ax is None:
        fig, ax = plot_one_var(xval, yval, yerr=yerrval, xname=xname, yname=yname, xscale=xscale, yscale=yscale,
                               label_name=label, legend=True,marker=marker)
    else:
        fig, ax = add_one_var_fig(fig, ax, xval, yval, yerr=yerrval, xscale=xscale, yscale=yscale,
                                  color=color, label_name=label, legend=True,marker=marker)



    # if isinstance(fname1, str):
    #     obj1 = read_obj(fname1)
    #     obj1.calc_avgs()
    #     obj1.calc_stds()
    #
    # if isinstance(fname2, str):
    #     obj2 = read_obj(fname2)
    #     obj2.calc_avgs()
    #     obj2.calc_stds()
    #
    # if isinstance(fname3, str):
    #     obj3 = read_obj(fname3)
    #     obj3.calc_avgs()
    #     obj3.calc_stds()
    #
    # if isinstance(fname4, str):
    #     obj4 = read_obj(fname4)
    #     obj4.calc_avgs()
    #     obj4.calc_stds()
    #
    # if isinstance(fname5, str):
    #     obj5 = read_obj(fname5)
    #     obj5.calc_avgs()
    #     obj5.calc_stds()
    #
    # xscale = 10 ** 9
    # # xscale = 1
    # # xname = "$\sigma \mathrm{[fT]}$"
    # # xname = "$\| SNR \|  \mathrm{[dB]}$"
    # xname = "$ \| q_{\mathrm{spont}} \|   \mathrm{[nAm]}$"
    #
    # # np.argwhere(np.array(obj.spontnoise[0]) == 0.0).tolist()
    # res = [idx for idx, val in enumerate(obj.noisestr[0]) if val == 0.0]
    #
    # fig1, ax1 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_avgcc)[res], yerr=np.array(obj.std_avgcc)[res], xname=xname, yname="$\mathrm{CC}$",
    #                          xscale=xscale, label_name=labels[0], legend=True)
    # fig2, ax2 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_avgre)[res], yerr=np.array(obj.std_avgre)[res], xname=xname, yname="$\mathrm{RE}$",
    #                          xscale=xscale, label_name=labels[0], legend=True)
    # fig3, ax3 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_avgdist)[res], yerr=np.array(obj.std_avgdist)[res], xname=xname,
    #                          yname="$d(p1, p2) \mathrm{[mm]}$", xscale=xscale, yscale=10 ** 3,
    #                          label_name=labels[0], legend=True)
    # fig4, ax4 = plot_one_var(np.array(obj.spontnoise[0])[res], np.array(obj.avg_snrdb)[res], yerr=np.array(obj.std_snrdb)[res], xname=xname, yname="$\mathrm{SNR [dB]}$",
    #                          xscale=xscale, label_name=labels[0], legend=True)
    #
    # if isinstance(fname1, str):
    #     add_one_var_fig(fig1, ax1, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_avgcc)[res], yerr=np.array(obj1.std_avgcc)[res], xscale=xscale, color="red",
    #                     label_name=labels[1], legend=True)
    #     add_one_var_fig(fig2, ax2, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_avgre)[res], yerr=np.array(obj1.std_avgre)[res], xscale=xscale, color="red",
    #                     label_name=labels[1], legend=True)
    #     add_one_var_fig(fig3, ax3, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_avgdist)[res], yerr=np.array(obj1.std_avgdist)[res], xscale=xscale,
    #                     yscale=10 ** 3, color="red", label_name=labels[1], legend=True)
    #     add_one_var_fig(fig4, ax4, np.array(obj1.spontnoise[0])[res], np.array(obj1.avg_snrdb)[res], yerr=np.array(obj1.std_snrdb)[res], xscale=xscale, color="red",
    #                     label_name=labels[1], legend=True)
    #
    # if isinstance(fname2, str):
    #     add_one_var_fig(fig1, ax1, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_avgcc)[res], yerr=np.array(obj2.std_avgcc)[res], xscale=xscale, color="blue",
    #                     label_name=labels[2], legend=True)
    #     add_one_var_fig(fig2, ax2, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_avgre)[res], yerr=np.array(obj2.std_avgre)[res], xscale=xscale, color="blue",
    #                     label_name=labels[2], legend=True)
    #     add_one_var_fig(fig3, ax3, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_avgdist)[res], yerr=np.array(obj2.std_avgdist)[res], xscale=xscale,
    #                     yscale=10 ** 3, color="blue", label_name=labels[2], legend=True)
    #     add_one_var_fig(fig4, ax4, np.array(obj2.spontnoise[0])[res], np.array(obj2.avg_snrdb)[res], yerr=np.array(obj2.std_snrdb)[res], xscale=xscale, color="blue",
    #                     label_name=labels[2], legend=True)
    #
    # if isinstance(fname3, str):
    #     add_one_var_fig(fig1, ax1, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_avgcc)[res], yerr=np.array(obj3.std_avgcc)[res], xscale=xscale, color="green",
    #                     label_name=labels[3], legend=True)
    #     add_one_var_fig(fig2, ax2, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_avgre)[res], yerr=np.array(obj3.std_avgre)[res], xscale=xscale, color="green",
    #                     label_name=labels[3], legend=True)
    #     add_one_var_fig(fig3, ax3, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_avgdist)[res], yerr=np.array(obj3.std_avgdist)[res], xscale=xscale,
    #                     yscale=10 ** 3, color="green", label_name=labels[3], legend=True)
    #     add_one_var_fig(fig4, ax4, np.array(obj3.spontnoise[0])[res], np.array(obj3.avg_snrdb)[res], yerr=np.array(obj3.std_snrdb)[res], xscale=xscale, color="green",
    #                     label_name=labels[3], legend=True)
    #
    # if isinstance(fname4, str):
    #     add_one_var_fig(fig1, ax1, obj4.spontnoise[0], obj4.avg_avgcc, yerr=obj4.std_avgcc, xscale=xscale, color="cyan",
    #                     label_name=labels[4], legend=True)
    #     add_one_var_fig(fig2, ax2, obj4.spontnoise[0], obj4.avg_avgre, yerr=obj4.std_avgre, xscale=xscale, color="cyan",
    #                     label_name=labels[4], legend=True)
    #     add_one_var_fig(fig3, ax3, obj4.spontnoise[0], obj4.avg_avgdist, yerr=obj4.std_avgdist, xscale=xscale,
    #                     yscale=10 ** 3, color="cyan", label_name=labels[4], legend=True)
    #     add_one_var_fig(fig4, ax4, obj4.spontnoise[0], obj4.avg_snrdb, yerr=obj4.std_snrdb, xscale=xscale, color="cyan",
    #                     label_name=labels[4], legend=True)
    #
    # if isinstance(fname5, str):
    #     add_one_var_fig(fig1, ax1, obj5.spontnoise[0], obj5.avg_avgcc, yerr=obj5.std_avgcc, xscale=xscale, color="pink",
    #                     label_name=labels[5], legend=True)
    #     add_one_var_fig(fig2, ax2, obj5.spontnoise[0], obj5.avg_avgre, yerr=obj5.std_avgre, xscale=xscale, color="pink",
    #                     label_name=labels[5], legend=True)
    #     add_one_var_fig(fig3, ax3, obj5.spontnoise[0], obj5.avg_avgdist, yerr=obj5.std_avgdist, xscale=xscale,
    #                     yscale=10 ** 3, color="pink", label_name=labels[5], legend=True)
    #     add_one_var_fig(fig4, ax4, obj5.spontnoise[0], obj5.avg_snrdb, yerr=obj5.std_snrdb, xscale=xscale, color="pink",
    #                     label_name=labels[5], legend=True)
    #
    # fig1.savefig(fname[:-4] + "-noisestr-cc.png", dpi=300)
    # fig2.savefig(fname[:-4] + "-noisestr-re.png", dpi=300)
    # fig3.savefig(fname[:-4] + "-noisestr-dist.png", dpi=300)
    # fig4.savefig(fname[:-4] + "-noisestr-snrdb.png", dpi=300)
    #
    # plt.show()

    return fig, ax


class AvgStatistics:
    def __init__(self):
        import numpy as np

        self.dist = []  # Array of all distances
        self.cc = []  # Array of all ccs
        self.re = []  # Array of all res
        self.rms_noise = []  # Array of all snrs
        self.rms_signal = []  # Array of all snrdbs

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
        self.rms_signal.append([])
        self.rms_noise.append([])

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

    def add_rmssignal(self, values_arr, subj_i):
        self.rms_signal[subj_i].append(values_arr)

    def add_rmsnoise(self, values_arr, subj_i):
        self.rms_noise[subj_i].append(values_arr)

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
        self.avg_rmssignal = np.average(self.rms_signal, axis=(0, 2))
        self.avg_rmsnoise = np.average(self.rms_noise, axis=(0, 2))

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
        self.std_rmssignal = np.std(self.rms_signal, axis=(0, 2))
        self.std_rmsnoise = np.std(self.rms_noise, axis=(0, 2))

    def save_obj(self, file_path):
        import pickle
        with open(file_path, 'wb') as output:
            pickle.dump(self, output)


def read_obj(file_path):
    import pickle
    with open(file_path, 'rb') as input:
        self = pickle.load(input)
    return self


def plot_one_var(x, y, yerr=[], xname="", yname="", show=False, xscale=1, yscale=1, savefig=False, label_name=None,
                 legend=False, font_size=14, marker='o'):
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

    ax.plot(x, y, linestyle='-', linewidth=2, c="black", markersize=7, marker=marker, label=label_name)
    if legend:
        ax.legend(fontsize=font_size)
    if len(yerr) > 0:
        yerr_temp = [yscale * i for i in yerr]
        ax.errorbar(x, y, yerr=yerr_temp, c="black", linestyle='none')

    ax.set_ylabel(yname, fontsize=font_size)
    ax.set_xlabel(xname, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()

    if isinstance(savefig, str):
        plt.savefig(savefig)

    if show == True:
        plt.show()

    return fig, ax


def add_one_var_fig(fig, ax, x, y, yerr=[], show=False, xscale=1, yscale=1, savefig=False, color="black",
                    label_name=None, legend=False, font_size=14, marker="o"):
    import matplotlib.pyplot as plt

    x = float(xscale) * np.array(x)
    y = yscale * y
    ax.plot(x, y, linestyle='-', linewidth=2, c=color, markersize=7, marker=marker, label=label_name)

    if len(yerr) > 0:
        yerr_temp = [yscale * i for i in yerr]
        ax.errorbar(x, y, yerr=yerr_temp, c=color, linestyle='none')

    if legend:
        ax.legend(fontsize=font_size)

    if show == True:
        fig.show()

    if isinstance(savefig, str):
        fig.savefig(savefig)

    return fig, ax


if __name__ == '__main__':
    main()
