import matplotlib.pyplot as plt
from Ska.Matplotlib import plot_cxctime, remake_ticks

R2A = 206264.8  # radians to arcsec


def fix_ylim(ax, min_ylim):
    y0, y1 = ax.get_ylim()
    dy = y1 - y0
    if dy < min_ylim:
        ymid = (y0 + y1) / 2.0
        y0 = ymid - min_ylim / 2.0
        y1 = ymid + min_ylim / 2.0
        ax.set_ylim(y0, y1)


def plot_manvr(manvr, figsize=(8, 10), fig=None):
    """
    Plot a Manvr event
    """
    ms = manvr.msidset
    if fig is None:
        fig = plt.figure(figsize=figsize)
    fig.clf()
    tstarts = [manvr.tstart, manvr.tstart]
    tstops = [manvr.tstop, manvr.tstop]
    colors = ['r', 'b', 'g', 'm']

    # AOATTQT estimated quaternion
    ax1 = fig.add_subplot(3, 1, 1)
    for i, color in zip(range(1, 5), colors):
        msid = 'aoattqt{}'.format(i)
        plot_cxctime(ms[msid].times, ms[msid].vals, fig=fig, ax=ax1,
                     color=color, linestyle='-', label=msid, interactive=False)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_title('Maneuver at {}'.format(manvr.start[:-4]))

    # AOATTER attitude error (arcsec)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    msids = ['aoatter{}'.format(i) for i in range(1, 4)] + ['one_shot']
    scales = [R2A, R2A, R2A, 1.0]
    for color, msid, scale in zip(colors, msids, scales):
        plot_cxctime(ms[msid].times, ms[msid].vals * scale, fig=fig, ax=ax2,
                     color=color, linestyle='-', label=msid, interactive=False)
    ax2.set_ylabel('Attitude error (arcsec)')
    fix_ylim(ax2, 10)

    # ACA sequence
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    msid = ms['aoacaseq']
    plot_cxctime(msid.times, msid.raw_vals, state_codes=msid.state_codes, fig=fig, ax=ax3,
                 interactive=False, label='aoacaseq')
    fix_ylim(ax3, 2.2)

    for ax in [ax1, ax2, ax3]:
        ax.callbacks.connect('xlim_changed', remake_ticks)
        plot_cxctime(tstarts, ax.get_ylim(), '--r', fig=fig, ax=ax, interactive=False)
        plot_cxctime(tstops, ax.get_ylim(), '--r', fig=fig, ax=ax, interactive=False)
        ax.grid()
        leg = ax.legend(loc='upper left', fontsize='small', fancybox=True)
        if leg is not None:
            leg.get_frame().set_alpha(0.7)

    fig.canvas.draw()
