import numpy as np
import pandas as pd
from scipy.optimize import linprog

NUCLEAR = 4300.0
COAL = 1000.0
LNG_CAP = 2100.0
LNG_MIN = 420.0
LNG_RAMP = 500.0
LNG_COST = 120.0
ESS_CAP = 5600.0
ESS_PWR = 1400.0
ESS_EFF = 0.95
SOC_MIN = 0.10
SOC_MAX = 0.90
SOC_INIT = 0.50
EXP_CAP = 3000.0
IMP_CAP = 3000.0
REWARD_SCALE = 500000.0
DAY_START = 4
DAY_END = 18
DAY_IMPORT_PENALTY = 350.0
DAY_IMPORT_CAP = 1600.0
TERMINAL_SOC_TARGET = 0.20
ESS_THROUGHPUT_COST = 4.0
IMPORT_PENALTY_BASE = 80.0
EXPORT_REVENUE_FACTOR = 0.45
EXPORT_TRANSACTION_COST = 45.0


def _soc_coeffs(T: int, t: int) -> np.ndarray:
    r = np.zeros(T * 5)
    for s in range(t + 1):
        r[T + s] = ESS_EFF / ESS_CAP
        r[2 * T + s] = -1.0 / (ESS_EFF * ESS_CAP)
    return r


def solve_mpc_plan(dm_fc, so_fc, wi_fc, imp_p, exp_p, init_soc, prev_lng, terminal_soc_target=TERMINAL_SOC_TARGET,
                   daytime_import_penalty=DAY_IMPORT_PENALTY, daytime_import_cap=DAY_IMPORT_CAP):
    T = len(dm_fc)
    N = T * 5
    c = np.zeros(N)
    for t in range(T):
        c[t] = LNG_COST
        c[T + t] += ESS_THROUGHPUT_COST
        c[2 * T + t] += ESS_THROUGHPUT_COST
        c[3 * T + t] = imp_p[t] + IMPORT_PENALTY_BASE
        export_credit = max(0.0, exp_p[t] * EXPORT_REVENUE_FACTOR - EXPORT_TRANSACTION_COST)
        c[4 * T + t] = -export_credit
        if DAY_START <= t <= DAY_END:
            c[3 * T + t] += daytime_import_penalty

    A_eq = np.zeros((T, N))
    b_eq = np.zeros(T)
    for t in range(T):
        net = dm_fc[t] - NUCLEAR - COAL - so_fc[t] - wi_fc[t]
        A_eq[t, t] = 1.0
        A_eq[t, T + t] = -1.0
        A_eq[t, 2 * T + t] = ESS_EFF
        A_eq[t, 3 * T + t] = 1.0
        A_eq[t, 4 * T + t] = -1.0
        b_eq[t] = net

    rows_ub = []
    b_ub = []

    for t in range(T):
        r = _soc_coeffs(T, t)
        rows_ub.append(r.copy())
        b_ub.append(SOC_MAX - init_soc)
        rows_ub.append(-r)
        b_ub.append(init_soc - SOC_MIN)

    end_soc = _soc_coeffs(T, T - 1)
    rows_ub.append(-end_soc)
    b_ub.append(init_soc - terminal_soc_target)

    for t in range(T):
        r = np.zeros(N)
        r[t] = 1.0
        if t == 0:
            rows_ub.append(r.copy())
            b_ub.append(prev_lng + LNG_RAMP)
            rows_ub.append(-r)
            b_ub.append(LNG_RAMP - prev_lng)
        else:
            r[t - 1] = -1.0
            rows_ub.append(r.copy())
            b_ub.append(LNG_RAMP)
            rows_ub.append(-r)
            b_ub.append(LNG_RAMP)

    bounds = []
    for _ in range(T):
        bounds.append((LNG_MIN, LNG_CAP))
    for _ in range(T):
        bounds.append((0, ESS_PWR))
    for _ in range(T):
        bounds.append((0, ESS_PWR))
    for t in range(T):
        cap = min(IMP_CAP, daytime_import_cap) if DAY_START <= t <= DAY_END else IMP_CAP
        bounds.append((0, cap))
    for _ in range(T):
        bounds.append((0, EXP_CAP))

    return linprog(c, A_ub=np.array(rows_ub), b_ub=np.array(b_ub),
                   A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')


def _forecast_slice(actual, forecast, start):
    rem = len(actual) - start
    out = np.array(forecast[start:start + rem], dtype=float)
    if rem > 0:
        out[0] = actual[start]
    return out


def run_rolling_mpc(profile, fc_demand, imp_p, exp_p, soc0=SOC_INIT, lng0=LNG_MIN,
                    terminal_soc_target=TERMINAL_SOC_TARGET, daytime_import_penalty=DAY_IMPORT_PENALTY,
                    daytime_import_cap=DAY_IMPORT_CAP):
    dm = np.asarray(profile['demand'], dtype=float)
    so = np.asarray(profile['solar'], dtype=float)
    wi = np.asarray(profile['wind'], dtype=float)
    so_fc_full = np.asarray(profile.get('solar_forecast', so), dtype=float)
    wi_fc_full = np.asarray(profile.get('wind_forecast', wi), dtype=float)

    soc = soc0
    lng_now = lng0
    rows = []

    for t in range(24):
        dm_fc = _forecast_slice(dm, fc_demand, t)
        so_fc = _forecast_slice(so, so_fc_full, t)
        wi_fc = _forecast_slice(wi, wi_fc_full, t)
        imp_fc = np.asarray(imp_p[t:], dtype=float)
        exp_fc = np.asarray(exp_p[t:], dtype=float)
        rem = len(dm_fc)
        target = max(terminal_soc_target, SOC_MIN) if rem <= 4 else terminal_soc_target

        res = solve_mpc_plan(dm_fc, so_fc, wi_fc, imp_fc, exp_fc, soc, lng_now,
                             terminal_soc_target=target,
                             daytime_import_penalty=daytime_import_penalty,
                             daytime_import_cap=daytime_import_cap)
        if not res.success:
            res = solve_mpc_plan(dm_fc, so_fc, wi_fc, imp_fc, exp_fc, soc, lng_now,
                                 terminal_soc_target=max(SOC_MIN, min(target, soc)),
                                 daytime_import_penalty=0.0,
                                 daytime_import_cap=IMP_CAP)
            if not res.success:
                raise RuntimeError(f'MPC solve failed at hour {t}: {res.message}')

        chg = max(0.0, min(res.x[rem], (SOC_MAX - soc) * ESS_CAP / ESS_EFF))
        soc_after_charge = soc + chg * ESS_EFF / ESS_CAP
        dis = max(0.0, min(res.x[2 * rem], (soc_after_charge - SOC_MIN) * ESS_CAP * ESS_EFF))
        soc = soc_after_charge - dis / (ESS_EFF * ESS_CAP)
        soc = float(np.clip(soc, SOC_MIN, SOC_MAX))

        lng_cmd = float(res.x[0])
        lng_now = float(np.clip(lng_cmd, lng_now - LNG_RAMP, lng_now + LNG_RAMP))
        lng_now = float(np.clip(lng_now, LNG_MIN, LNG_CAP))

        ess_net = chg - dis * ESS_EFF
        fixed = NUCLEAR + COAL + so[t] + wi[t]
        total = fixed + lng_now - ess_net
        surplus = max(0.0, total - dm[t])
        export_mwh = min(EXP_CAP, surplus)
        deficit = max(0.0, dm[t] - (total - export_mwh))
        grid_import = min(deficit, IMP_CAP)
        curtailment = max(0.0, (total - export_mwh) - dm[t] - grid_import)
        export_credit = max(0.0, exp_p[t] * EXPORT_REVENUE_FACTOR - EXPORT_TRANSACTION_COST)
        cost = lng_now * LNG_COST + grid_import * imp_p[t] - export_mwh * export_credit + (chg + dis) * ESS_THROUGHPUT_COST

        rows.append({
            'nuclear': NUCLEAR,
            'coal': COAL,
            'lng': lng_now,
            'solar': so[t],
            'wind': wi[t],
            'demand': dm[t],
            'grid_import': grid_import,
            'export_mwh': export_mwh,
            'curtailment': curtailment,
            'soc': soc,
            'ess_charge': chg,
            'ess_discharge': dis,
            'cost': cost,
            'reward': -cost / REWARD_SCALE,
        })
    return pd.DataFrame(rows)
