"""D235 fixtures with explicit feedback and pre-existing source-gauge holds."""

from rbpf_move_cases import make_step_case


def make_feedback_case(count=64,unique=8,*,feedback=True,preload=True,**options):
    runtime,rtk,ctx = make_step_case(count,unique,feedback=feedback,**options)
    if feedback and preload:
        # A source fixture's first nonempty H gives consistent held SD values.
        # Both variants receive identical ordered committed/nav state. No
        # candidates, priors, thresholds, particles or numerical method changes.
        h = next((p.H for p in runtime.particles if p.H),{})
        for group,integers in h.items():
            for sat,z in integers.items():
                key = (int(sat),group[1])
                rtk._committed[key] = float(z)
                rtk.nav.x[rtk.IB(key[0],key[1],rtk.nav.na)] = float(z)
    return runtime,rtk,ctx
