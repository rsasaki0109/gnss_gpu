/*------------------------------------------------------------------------------
 * rtklib_spp.c: RTKLIB pntpos SPP measurement export (library version)
 *
 * This is the library equivalent of export_spp_meas.c.  Instead of fprintf to
 * stdout it appends measurements to a dynamically-grown array so that the
 * caller (e.g. pybind11 wrapper) can consume the data directly.
 *
 * The helper functions (varerr_spp, gettgd_m, prange_gps_l1) and the RTKLIB
 * stub functions (showmsg, settspan, settime) are identical to the ones in
 * export_spp_meas.c.
 *-----------------------------------------------------------------------------*/
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rtklib.h"
#include "rtklib_spp/rtklib_spp.h"

/* ---- RTKLIB stubs (required when linking postpos.o etc.) ---- */

int showmsg(const char *format, ...)
{
    (void)format;
    return 0;
}

void settspan(gtime_t ts, gtime_t te)
{
    (void)ts;
    (void)te;
}

void settime(gtime_t time)
{
    (void)time;
}

/* ---- internal helpers (same logic as export_spp_meas.c) ---- */

#define ERR_CBIAS 0.3
#define SQR(x) ((x) * (x))
#define INIT_CAPACITY 4096

static double varerr_spp(double el, int sys)
{
    double fact = 1.0, varr;
    double el_min = 5.0 * D2R;
    (void)sys;
    if (el < el_min) el = el_min;
    varr = SQR(0.003) + SQR(0.003) / sin(el);
    varr *= SQR(300.0);
    return SQR(fact) * varr;
}

static double gettgd_m(int sat, const nav_t *nav)
{
    int i, sys = satsys(sat, NULL);

    if (sys == SYS_GLO) {
        for (i = 0; i < nav->ng; i++) {
            if (nav->geph[i].sat == sat) break;
        }
        return (i >= nav->ng) ? 0.0 : -nav->geph[i].dtaun * CLIGHT;
    }
    for (i = 0; i < nav->n; i++) {
        if (nav->eph[i].sat == sat) break;
    }
    return (i >= nav->n) ? 0.0 : nav->eph[i].tgd[0] * CLIGHT;
}

static double prange_gps_l1(const obsd_t *obs, const nav_t *nav, double *vmeas)
{
    int sat = obs->sat, sys = satsys(sat, NULL), bias_ix;
    double P1 = obs->P[0];

    *vmeas = SQR(ERR_CBIAS);
    if (P1 == 0.0) return 0.0;
    bias_ix = code2bias_ix(sys, obs->code[0]);
    if (bias_ix > 0) P1 += nav->cbias[sat - 1][0][bias_ix - 1];
    if (sys == SYS_GPS || sys == SYS_QZS) {
        return P1 - gettgd_m(sat, nav);
    }
    return 0.0;
}

/* ---- result buffer management ---- */

static int result_push(rtklib_spp_result_t *res, const rtklib_spp_meas_t *m)
{
    if (res->n_meas >= res->capacity) {
        int newcap = res->capacity == 0 ? INIT_CAPACITY : res->capacity * 2;
        rtklib_spp_meas_t *tmp = (rtklib_spp_meas_t *)realloc(
            res->meas, (size_t)newcap * sizeof(rtklib_spp_meas_t));
        if (!tmp) return -1;
        res->meas = tmp;
        res->capacity = newcap;
    }
    res->meas[res->n_meas++] = *m;
    return 0;
}

/* ---- public API ---- */

int rtklib_spp_export(const char *obs_file, const char *nav_file,
                      double el_mask_deg, rtklib_spp_result_t *result)
{
    obs_t obs = {0};
    nav_t nav = {0};
    sta_t sta = {{0}};
    prcopt_t opt = prcopt_default;
    char msg[128];
    int i, j, m, week;
    double tow;

    if (!result) return -1;
    memset(result, 0, sizeof(*result));

    if (!readrnx(obs_file, 1, "", &obs, &nav, &sta)) {
        return -1;
    }
    if (!readrnx(nav_file, 1, "", &obs, &nav, &sta)) {
        freeobs(&obs);
        return -1;
    }
    uniqnav(&nav);
    sortobs(&obs);

    opt.mode   = PMODE_SINGLE;
    opt.navsys = SYS_GPS;
    opt.nf     = 1;
    opt.ionoopt = IONOOPT_BRDC;
    opt.tropopt = TROPOPT_SAAS;
    opt.sateph  = EPHOPT_BRDC;
    opt.elmin   = el_mask_deg * D2R;

    for (i = 0; i < obs.n;) {
        int rcv = obs.data[i].rcv;
        gtime_t t0 = obs.data[i].time;
        j = i;
        while (j < obs.n && obs.data[j].rcv == rcv &&
               timediff(obs.data[j].time, t0) <= DTTOL)
            j++;
        m = j - i;
        if (m <= 0) { i = j; continue; }

        {
            obsd_t *oe = obs.data + i;
            sol_t sol = {{0}};
            double *rs, *dts, *var, azel[2 * MAXOBS];
            int svh[MAXOBS];
            ssat_t ssat[MAXSAT];
            double rr[3], pos[3], e[3];
            int k, sat, sys, stat;

            sol.time = oe[0].time;
            tow = time2gpst(oe[0].time, &week);

            for (k = 0; k < MAXSAT; k++) ssat[k] = (ssat_t){0};

            rs  = mat(6, m);
            dts = mat(2, m);
            var = mat(1, m);
            satposs(sol.time, oe, m, &nav, opt.sateph, rs, dts, var, svh);

            stat = pntpos(oe, m, &nav, &opt, &sol, azel, ssat, msg);
            if (!stat) {
                free(rs); free(dts); free(var);
                i = j;
                continue;
            }

            for (k = 0; k < 3; k++) rr[k] = sol.rr[k];
            ecef2pos(rr, pos);

            for (k = 0; k < m; k++) {
                double r, dion = 0, dtrp = 0, vion, vtrp, P, vmeas, azk[2];
                rtklib_spp_meas_t meas;

                if (k < m - 1 && oe[k].sat == oe[k + 1].sat) continue;
                sat = oe[k].sat;
                if (!(sys = satsys(sat, NULL)) || sys != SYS_GPS) continue;
                if (!ssat[sat - 1].vs) continue;
                if (satexclude(sat, var[k], svh[k], &opt)) continue;

                if ((r = geodist(rs + 6 * k, rr, e)) <= 0.0) continue;
                if (satazel(pos, e, azk) < opt.elmin) continue;

                if (testsnr(0, 0, azk[1], oe[k].SNR[0], &opt.snrmask)) continue;

                if (!ionocorr(oe[k].time, &nav, sat, pos, azk,
                              opt.ionoopt, &dion, &vion))
                    continue;
                {
                    double freq = sat2freq(sat, oe[k].code[0], &nav);
                    if (freq == 0.0) continue;
                    dion *= SQR(FREQL1 / freq);
                }
                if (!tropcorr(oe[k].time, &nav, pos, azk,
                              opt.tropopt, &dtrp, &vtrp))
                    continue;

                if ((P = prange_gps_l1(oe + k, &nav, &vmeas)) == 0.0) continue;

                memset(&meas, 0, sizeof(meas));
                meas.gps_week  = week;
                meas.gps_tow   = tow;
                meas.prn       = sat;
                satno2id(sat, meas.sat_id);
                meas.prange_m  = P;
                meas.r_m       = r;
                meas.iono_m    = dion;
                meas.trop_m    = dtrp;
                meas.sat_clk_m = -CLIGHT * dts[k * 2];
                meas.satx      = rs[k * 6];
                meas.saty      = rs[1 + k * 6];
                meas.satz      = rs[2 + k * 6];
                meas.el_rad    = azk[1];
                {
                    double varerr_val = varerr_spp(azk[1], sys);
                    meas.var_total = var[k] + vmeas + vion + vtrp + varerr_val;
                }

                if (result_push(result, &meas) != 0) {
                    free(rs); free(dts); free(var);
                    freeobs(&obs);
                    freenav(&nav, 0xFF);
                    return -1;  /* allocation failure */
                }
            }

            free(rs); free(dts); free(var);
        }
        i = j;
    }

    freeobs(&obs);
    freenav(&nav, 0xFF);
    return 0;
}

void rtklib_spp_free(rtklib_spp_result_t *result)
{
    if (result) {
        free(result->meas);
        result->meas = NULL;
        result->n_meas = 0;
        result->capacity = 0;
    }
}
