"""Final numerical certificate reconstruction and verification.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

import sys

sys.dont_write_bytecode = True
from gate1_common import *
import scipy.sparse as sp
from fractions import Fraction


def main(year):
    folder = GATE / f"year_{year}"
    z = np.load(folder / "MODEL_ARRAYS.npz", allow_pickle=False)
    v = np.load(folder / "FULL_SOLVER_VECTORS.npz", allow_pickle=False)
    Ax = sp.load_npz(folder / "A_support.npz")
    D = sp.load_npz(folder / "D_budget.npz")
    ident = read(folder / "MODEL_IDENTITY.json")
    solver = read(folder / "SOLVER_RESULT.json")
    T = len(z["dt"])
    RR = len(z["upper"])
    hh = z["support_hours"]
    bb = z["support_intercepts"]
    upper = z["upper"]
    H = z["allowance"]
    nx = z["variable_positions"]
    tx = z["variable_hours"]
    assert (
        v["budget_pi"].shape == (114,)
        and v["support_pi"].shape == (len(hh),)
        and v["x"].shape == (RR,)
    )
    lam_raw = -v["budget_pi"]
    zeta_raw = -v["support_pi"]
    lam = np.maximum(lam_raw, 0)
    zet = np.maximum(zeta_raw, 0)
    sums = np.bincount(hh, weights=zet, minlength=T)
    scale = np.ones(T)
    hit = sums > 1 - 2**-35
    scale[hit] = (1 - 2**-35) / sums[hit]
    zet *= scale[hh]
    exact_max = Fraction(0)
    hour_violations = 0
    for t in np.unique(hh):
        s = fraction_sum(zet[hh == t])
        exact_max = max(exact_max, s)
        hour_violations += int(s > 1)
    assert hour_violations == 0 and np.all(lam >= 0) and np.all(zet >= 0)
    np.savez_compressed(
        folder / "ADMISSIBLE_DUAL_MULTIPLIERS.npz",
        lambda_budget=lam,
        zeta_support=zet,
        support_hour=hh,
        lambda_raw=lam_raw,
        zeta_raw=zeta_raw,
        hour_scale=scale,
    )
    r = np.asarray(D.T @ lam + Ax.T @ zet).ravel()
    allowance_term = -math.fsum(float(h) * float(l) for h, l in zip(H, lam))
    intercept_term = math.fsum(float(a) * float(b) for a, b in zip(zet, bb))
    box_term = math.fsum(float(b) * min(float(q), 0.0) for b, q in zip(upper, r))
    L0 = math.fsum([allowance_term, intercept_term, box_term])
    # Quantified numerical guard. It includes support formation/input-to-affine
    # error bounds, sparse accumulation, quotient/budget and scalar reductions.
    plane_guard = math.fsum(
        float(a) * float(e) for a, e in zip(zet, z["support_plane_error_bounds"])
    )
    kcol = int(np.max(np.diff(Ax.tocsc().indptr), initial=0)) + 5
    magnitude_r = np.asarray(abs(Ax).T @ zet + abs(D).T @ lam).ravel()
    r_error = gamma(kcol) * magnitude_r + gamma(3) * np.asarray(abs(D).T @ lam).ravel()
    stationarity_guard = math.fsum(float(b) * float(e) for b, e in zip(upper, r_error))
    sum_magnitudes = (
        math.fsum(abs(float(h) * float(l)) for h, l in zip(H, lam))
        + math.fsum(abs(float(a) * float(b)) for a, b in zip(zet, bb))
        + math.fsum(abs(float(b) * min(float(q), 0.0)) for b, q in zip(upper, r))
    )
    scalar_guard = gamma(6) * sum_magnitudes + gamma(3) * (
        abs(allowance_term) + abs(intercept_term) + abs(box_term)
    )
    guard = np.nextafter(
        (plane_guard + stationarity_guard + scalar_guard) * (1 + gamma(32)), np.inf
    )
    L = float(np.nextafter(L0 - guard, -np.inf))
    # Independently recompute the certificate via aggregated signed physical rows,
    # not the constructed support coefficient matrix.
    data = physical()
    base = np.load(folder / "requested_flow.npy", mmap_mode="r")
    sel = np.load(folder / "ARCHIVED_SUPPORT_SELECTORS.npz", allow_pickle=False)
    po, no, pr, nr = [sel[k] for k in ["pos_offsets", "neg_offsets", "pos_rows", "neg_rows"]]
    byhour = [np.flatnonzero(hh == t) for t in range(T)]
    rr = np.asarray(D.T @ lam).ravel()
    row_intercepts = []
    wp_rows = []
    wn_rows = []
    wp_cols = []
    wn_cols = []
    wp_data = []
    wn_data = []
    for t, indices in enumerate(byhour):
        if not len(indices):
            continue
        wp = np.zeros(len(data["F"]))
        wn = np.zeros(len(data["F"]))
        for k in indices:
            if zet[k]:
                np.add.at(wp, pr[po[k] : po[k + 1]], zet[k])
                np.add.at(wn, nr[no[k] : no[k + 1]], zet[k])
        pn = np.flatnonzero(wp)
        nn = np.flatnonzero(wn)
        wp_rows.extend([t] * len(pn))
        wp_cols.extend(pn)
        wp_data.extend(wp[pn])
        wn_rows.extend([t] * len(nn))
        wn_cols.extend(nn)
        wn_data.extend(wn[nn])
        grad = (wp - wn) @ data["G"]
        cols = np.flatnonzero(tx == t)
        rr[cols] += grad[nx[cols]]
        row_intercepts.append(
            math.fsum(
                [float(wp[j]) * (float(base[t, j]) - float(data["F"][j])) for j in pn]
                + [float(wn[j]) * (-float(base[t, j]) - float(data["F"][j])) for j in nn]
            )
        )
    row_b = math.fsum(row_intercepts)
    row_box = math.fsum(float(b) * min(float(q), 0) for b, q in zip(upper, rr))
    Lrow = math.fsum([allowance_term, row_b, row_box])
    for name, rows, cols, vals in [
        ("positive", wp_rows, wp_cols, wp_data),
        ("negative", wn_rows, wn_cols, wn_data),
    ]:
        sp.save_npz(
            folder / f"AGGREGATED_{name.upper()}_ROW_DUALS.npz",
            sp.csr_matrix((vals, (rows, cols)), shape=(T, len(data["F"]))),
        )
    # The actual master primal/dual residuals and finite-box bound identity.
    budget_violation = float(np.maximum(D @ v["x"] - H, 0).max())
    support_violation = float(np.maximum(Ax @ v["x"] + bb - v["theta"][hh], 0).max())
    bound_violation = max(
        0, float((-v["x"]).max()), float((v["x"] - upper).max()), float((-v["theta"]).max())
    )
    rawr = np.asarray(D.T @ lam_raw + Ax.T @ zeta_raw).ravel()
    rawdual = math.fsum(
        [-float(H @ lam_raw), float(bb @ zeta_raw), float(upper @ np.minimum(rawr, 0))]
    )
    gap = solver["objective"] - L
    ix = np.flatnonzero(data["dates"].year == year)
    xb = data["xbar_mw"][ix]
    m2folder = R / f"annual_primary/year_{year}/rho_0.15/annual/M2"
    m2info = read(m2folder / "solution.json")
    p = register(m2folder / "solution_arrays.npz")
    assert sha(p) == m2info["arrays_sha256"]
    m2 = np.load(p, allow_pickle=False)["x_mw"]
    b03 = read(R / f"b03_certificates/year_{year}/B03_M1_M2_CERTIFICATE.json")
    factors = np.array(b03["witness_scaling_factors"])
    witness = m2 * factors
    assert sha_arr(witness) == b03["witness_hash"]
    assert (
        np.all((factors >= 0) & (factors <= 1)) and np.all(witness >= 0) and np.all(witness <= xb)
    )
    exact_debits = [
        fraction_sum(witness[:, n]) / Fraction(float(P))
        for n, P in enumerate(data["capacities_mw"])
    ]
    assert all(a <= Fraction(float(h)) for a, h in zip(exact_debits, H))
    residuals = []
    peak = 0.0
    peakdata = {}
    U_error_components = []
    for start in range(0, T, 128):
        stop = min(T, start + 128)
        req = (
            data["p_background_mw"][ix[start:stop]]
            - data["requests_mw"][ix[start:stop]] @ data["A"].T
        )
        # Source operation path p_req+A*x thenH, independently of cached base+G*x.
        flow = (req + witness[start:stop] @ data["A"].T) @ data["H"].T + data["h"]
        over = np.maximum(np.abs(flow) - data["F"], 0)
        residuals.append(float(over.sum()))
        idx = np.unravel_index(over.argmax(), over.shape)
        if over[idx] > peak:
            peak = float(over[idx])
            peakdata = {
                "timestamp": str(data["dates"][ix[start + idx[0]]]),
                "row": data["row_ids"][idx[1]],
            }
        # A forward-error diagnostic; not a formal interval enclosure of the complete input/model pipeline.
        absreq = (
            np.abs(data["p_background_mw"][ix[start:stop]])
            + np.abs(data["requests_mw"][ix[start:stop]]) @ np.abs(data["A"]).T
        )
        ep = gamma(120) * absreq + gamma(120) * (witness[start:stop] @ np.abs(data["A"]).T)
        ef = ep @ np.abs(data["H"]).T + gamma(50) * (
            (np.abs(req) + witness[start:stop] @ np.abs(data["A"]).T) @ np.abs(data["H"]).T
            + np.abs(data["h"])
        )
        # Only potentially positive hinges contribute to upper evaluation uncertainty.
        near = np.abs(flow) - data["F"] >= -ef
        U_error_components.append(float(ef[near].sum()) + gamma(over.size + 4) * float(over.sum()))
    U = math.fsum(residuals)
    Ustored = float(b03["witness_S_mw_h_row"])
    Uerr = math.fsum(U_error_components) + gamma(len(residuals) + 3) * abs(U)
    assert abs(U - Ustored) <= max(Uerr, 1e-6)
    delta = max(1e-6, 1e-9 * max(1, abs(Ustored), abs(L)))
    margin = L - Ustored - delta
    robustmargin = margin - Uerr
    relative = 100 * (1 - (Ustored + delta) / L)
    verdict = (
        "CONDITIONAL PASS"
        if solver["solver_status"] == "OPTIMAL"
        and margin > 0
        and robustmargin > 0
        and abs(Lrow - L0) <= max(guard, 1e-5)
        else "FAIL"
    )
    dump(
        folder / "INDEPENDENT_CERTIFICATE.json",
        {
            "year": year,
            "verdict": verdict,
            "precision_class": "Guarded floating-point numerical certificate with complete exported multipliers; not a fully outward-rounded interval certificate.",
            "new_lower_bound": L,
            "unguarded_reconstructed_bound": L0,
            "numerical_downward_guard": float(guard),
            "support_plane_formation_guard": plane_guard,
            "stationarity_accumulation_guard": stationarity_guard,
            "scalar_reduction_guard": scalar_guard,
            "budget_term": allowance_term,
            "support_intercept_term": intercept_term,
            "finite_box_correction": box_term,
            "raw_solver_dual_reconstruction": rawdual,
            "solver_primal_objective": solver["objective"],
            "new_primal_dual_gap": gap,
            "independent_row_aggregation_bound": Lrow,
            "independent_row_vs_support_difference": Lrow - L0,
            "stationarity_path_max_difference": float(np.max(np.abs(r - rr))),
            "max_exact_hour_multiplier_sum": float(exact_max),
            "exact_hour_sum_violations": hour_violations,
            "budget_sign_projection_max": float(np.max(np.abs(lam - lam_raw))),
            "support_projection_max": float(np.max(np.abs(zet - zeta_raw))),
            "primal_budget_violation_h": budget_violation,
            "primal_support_violation": support_violation,
            "primal_bound_violation_MW": bound_violation,
            "M2_saved_witness_identity": True,
            "M2_exact_saved_coordinate_budget_pass": True,
            "M2_exact_saved_coordinate_box_pass": True,
            "M2_upper_archived": Ustored,
            "M2_replayed": U,
            "M2_replay_difference": U - Ustored,
            "M2_replay_forward_error_estimate": Uerr,
            "delta": delta,
            "separation_margin": margin,
            "margin_after_additional_upper_replay_error": robustmargin,
            "relative_percent": relative,
            "historical_lower_bound": b03["m1lp_certified_lower_bound"],
            "historical_vs_new_lower_difference": L - b03["m1lp_certified_lower_bound"],
            "conditions": [
                "Fixed frozen numerical model and standard floating-point forward-error analysis; all operations are not formally outward interval enclosed.",
                "Archived M2 upper replay is a numerical endpoint; the replay-error estimate is separately shown and does not close the large separation.",
            ],
            "optimizer_calls_in_verifier": 0,
        },
    )
    with (folder / "POSITIONWISE_M2_WITNESS_DEBIT.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "position",
                "H_hours",
                "debit_float_h",
                "exact_debit_numerator",
                "exact_debit_denominator",
                "exact_feasible",
            ]
        )
        writer.writerows(
            (pid, h, float(e), e.numerator, e.denominator, e <= Fraction(float(h)))
            for pid, h, e in zip(data["meta"]["participant_ids"], H, exact_debits)
        )
    print(
        "CERTIFICATE",
        year,
        verdict,
        "L",
        L,
        "U",
        Ustored,
        "MARGIN",
        margin,
        "ROBUST_MARGIN",
        robustmargin,
        "GUARD",
        guard,
        flush=True,
    )
    if verdict == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main(int(sys.argv[1]))
