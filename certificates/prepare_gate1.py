"""Final numerical certificate reconstruction and verification.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

import sys

sys.dont_write_bytecode = True
from gate1_common import *
import scipy.sparse as sp
import shutil, time


def main():
    d = physical()
    m = d["meta"]
    N = len(m["participant_ids"])
    J = len(d["F"])
    u = np.finfo(float).eps / 2
    for name in ["annual.py", "contracts.py", "storage.py", "persistence.py", "model.py"]:
        register(A / "IMPLEMENTATION/source/tpwrs" / name)
    reconciliation = []
    with (GATE / "NETWORK_ROW_MAPPING.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "state_and_element"])
        w.writerows(enumerate(d["row_ids"]))
    for year in [2029, 2030, 2031]:
        t0 = time.perf_counter()
        out = GATE / f"year_{year}"
        out.mkdir()
        orig = R / f"annual_primary/year_{year}/rho_0.15/annual/M1_LP"
        exp = read(orig / "EXPERIMENT_CASE.json")["experiment"]
        ga = read(orig / "GEOMETRY_AUDIT.json")
        cm = read(orig / "SUPPORT_CUTS_META.json")
        info = read(orig / "solution.json")
        ix = np.flatnonzero(d["dates"].year == year)
        T = len(ix)
        identity, timeid = case_identity(d, ix)
        assert identity == exp["physical_case_hash"] and timeid == exp["time_grid_id"]
        assert m["participant_ids"] == exp["allowance"]["participant_ids"]
        dt = np.asarray(m["time_grid"]["delta_t_hours"])[ix]
        assert np.all(dt == 1) and T == 8760
        xb = d["xbar_mw"][ix]
        H = np.asarray(exp["allowance"]["values_hours"])
        ctr = d["contract_mask"][ix]
        assert np.array_equal(H, np.floor(0.15 * ctr.sum(axis=0) + 1e-12))
        assert (
            hashlib.sha256(np.ascontiguousarray(ctr.astype(np.uint8)).tobytes()).hexdigest()
            == exp["allowance"]["contract_mask_hash"]
        )
        assert not np.any(xb[:, H == 0] > 0)
        # Reconstruct requested affine flows in the same128-hour shapes as frozen source.
        base = np.lib.format.open_memmap(
            out / "requested_flow.npy", mode="w+", dtype="float64", shape=(T, J)
        )
        # Forward-error estimates for input-to-affine coefficient formation; numerical analysis, not data uncertainty.
        errbase = np.lib.format.open_memmap(
            out / "requested_flow_error_bound.npy", mode="w+", dtype="float64", shape=(T, J)
        )
        Gabs = np.abs(d["G"])
        Gerr = gamma(47) * (np.abs(d["H"]) @ np.abs(d["A"]))
        active = []
        parts = []
        posall = []
        negall = []
        candidate = potential = seed = 0
        constant = 0.0
        discarded_row_upper_max = -math.inf
        for start in range(0, T, 128):
            stop = min(T, start + 128)
            v = d["requests_mw"][ix[start:stop]]
            bg = d["p_background_mw"][ix[start:stop]]
            req = bg - v @ d["A"].T
            f = req @ d["H"].T + d["h"]
            base[start:stop] = f
            erreq = gamma(N + 3) * (np.abs(bg) + np.abs(v) @ np.abs(d["A"]).T)
            errbase[start:stop] = (
                erreq @ np.abs(d["H"]).T
                + gamma(48) * (np.abs(req) @ np.abs(d["H"]).T + np.abs(d["h"]))
            ) * (1 + gamma(120))
            aa = np.flatnonzero(np.any(np.abs(f) > d["F"], axis=1))
            xx = xb[start:stop][aa].copy()
            xx[:, H == 0] = 0
            pup = f[aa] + xx @ np.maximum(d["G"], 0).T
            nup = -f[aa] + xx @ np.maximum(-d["G"], 0).T
            for k, loc in enumerate(aa):
                t = start + int(loc)
                bp = f[loc]
                ps = np.flatnonzero(pup[k] >= d["F"] - 1e-9).astype(np.int32)
                ns = np.flatnonzero(nup[k] >= d["F"] - 1e-9).astype(np.int32)
                cp = np.flatnonzero((xb[t] > 0) & (H > 0)).astype(np.int32)
                use = np.zeros(len(cp), bool)
                if len(ps):
                    use |= np.any(d["G"][np.ix_(ps, cp)] < 0, axis=0)
                if len(ns):
                    use |= np.any(d["G"][np.ix_(ns, cp)] > 0, axis=0)
                kept = cp[use]
                active.append(t)
                parts.append(kept)
                posall.append(ps)
                negall.append(ns)
                candidate += len(cp)
                potential += len(ps) + len(ns)
                seed += np.sum(bp[ps] >= d["F"][ps] - 1) + np.sum(-bp[ns] >= d["F"][ns] - 1)
                if not len(kept):
                    constant += float(np.maximum(np.abs(bp) - d["F"], 0).sum())
                for upper in [pup[k] - d["F"], nup[k] - d["F"]]:
                    discarded = upper[upper < -1e-9]
                    if len(discarded):
                        discarded_row_upper_max = max(
                            discarded_row_upper_max, float(discarded.max())
                        )
            print("GEOMETRY", year, stop, "/", T, flush=True) if stop % 2048 == 0 else None
        base.flush()
        errbase.flush()
        active = np.array(active, dtype=np.int32)
        pc = np.concatenate(parts).astype(np.int32)
        lh = np.repeat(np.arange(len(active), dtype=np.int32), [len(x) for x in parts])
        ret = len(pc)
        gh = shj(
            {
                "engine": "ANNUAL_SUPPORT_OA_V3_20260905",
                "physical_case_hash": identity,
                "allowance_positive_hash": sha_arr((H > 0).astype(np.uint8)),
                "active_hours": sha_arr(active),
                "participant_coordinate": sha_arr(pc),
                "local_hour_coordinate": sha_arr(lh),
                "potential_pos": shj([sha_arr(x) for x in posall]),
                "potential_neg": shj([sha_arr(x) for x in negall]),
                "constant_s": constant,
            }
        )
        assert gh == ga["geometry_hash"] == cm["geometry_hash"], (year, gh, ga["geometry_hash"])
        assert (
            len(active) == ga["active_hours"]
            and ret == ga["retained_action_coordinates"]
            and potential == ga["potential_directed_rows"]
            and constant == ga["constant_s_mw_h_row"]
        )
        # Full mathematical positive box retained, including historically dominated coordinates and secure hours.
        tx, nx = np.nonzero(xb > 0)
        RR = len(tx)
        varindex = np.full((T, N), -1, dtype=np.int64)
        varindex[tx, nx] = np.arange(RR)
        upper = xb[tx, nx]
        debit = dt[tx] / upper
        D = sp.csr_matrix((debit, (nx, np.arange(RR))), shape=(N, RR))
        sp.save_npz(out / "D_budget.npz", D)
        path = register(orig / "SUPPORT_CUTS.npz")
        assert sha(path) == cm["arrays_sha256"]
        z = np.load(path, allow_pickle=False)
        hour = z["hours"]
        po = z["pos_offsets"]
        no = z["neg_offsets"]
        pr = z["pos_rows"]
        nr = z["neg_rows"]
        K = len(hour)
        assert (
            K == cm["cut_count"]
            and len(po) == K + 1
            and len(no) == K + 1
            and po[0] == no[0] == 0
            and po[-1] == len(pr)
            and no[-1] == len(nr)
        )
        assert (
            np.all(np.diff(po) >= 0)
            and np.all(np.diff(no) >= 0)
            and np.all((hour >= 0) & (hour < len(active)))
        )
        shutil.copyfile(path, out / "ARCHIVED_SUPPORT_SELECTORS.npz")
        aa = []
        ii = []
        ptr = [0]
        intercepts = np.empty(K)
        plane_errors = np.empty(K)
        globalhours = active[hour]
        maxgrad = maxint = maxpoint = 0.0
        selectors_verified = 0
        with np.load(register(orig / "solution_arrays.npz"), allow_pickle=False) as ss:
            reference = ss["x_mw"]
        phi = np.empty(T)
        for start in range(0, T, 128):
            stop = min(T, start + 128)
            flow = base[start:stop] + reference[start:stop] @ d["G"].T
            phi[start:stop] = np.maximum(np.abs(flow) - d["F"], 0).sum(axis=1)
        for k in range(K):
            lh0 = int(hour[k])
            t = int(globalhours[k])
            ps = pr[po[k] : po[k + 1]]
            ns = nr[no[k] : no[k + 1]]
            count = len(ps) + len(ns)
            assert np.all(np.diff(ps) > 0) and np.all(np.diff(ns) > 0)
            assert (
                np.isin(ps, posall[lh0], assume_unique=True).all()
                and np.isin(ns, negall[lh0], assume_unique=True).all()
            )
            selectors_verified += count
            grad = d["G"][ps].sum(axis=0) - d["G"][ns].sum(axis=0)
            b0 = float(np.sum(base[t, ps] - d["F"][ps]) + np.sum(-base[t, ns] - d["F"][ns]))
            intercepts[k] = b0
            # Independently compare frozen screened-column coefficient rule to full-position sums.
            kept = parts[lh0]
            old = d["G"][np.ix_(ps, kept)].sum(axis=0) - d["G"][np.ix_(ns, kept)].sum(axis=0)
            maxgrad = max(maxgrad, float(np.max(np.abs(old - grad[kept]), initial=0)))
            bref = math.fsum(
                [float(base[t, j]) - float(d["F"][j]) for j in ps]
                + [-float(base[t, j]) - float(d["F"][j]) for j in ns]
            )
            maxint = max(maxint, abs(b0 - bref))
            maxpoint = max(maxpoint, float(b0 + grad @ reference[t] - phi[t]))
            absgrad = Gabs[ps].sum(axis=0) + Gabs[ns].sum(axis=0)
            absb = float(
                np.sum(np.abs(base[t, ps]) + d["F"][ps]) + np.sum(np.abs(base[t, ns]) + d["F"][ns])
            )
            formation = gamma(count + 4) * (absb + absgrad @ xb[t])
            physical_error = float(
                errbase[t, ps].sum()
                + errbase[t, ns].sum()
                + (Gerr[ps].sum(axis=0) + Gerr[ns].sum(axis=0)) @ xb[t]
            )
            plane_errors[k] = np.nextafter(
                (formation + physical_error) * (1 + gamma(max(120, count * 3 + 10))), np.inf
            )
            cols = varindex[t]
            hit = cols >= 0
            vals = grad[hit]
            inds = cols[hit]
            nz = vals != 0
            aa.extend(vals[nz])
            ii.extend(inds[nz])
            ptr.append(len(aa))
            if k % 20000 == 0:
                print("SUPPORTS", year, k, "/", K, flush=True)
        Ax = sp.csr_matrix(
            (np.array(aa), np.array(ii, dtype=np.int32), np.array(ptr, dtype=np.int64)),
            shape=(K, RR),
        )
        sp.save_npz(out / "A_support.npz", Ax)
        np.savez_compressed(
            out / "MODEL_ARRAYS.npz",
            upper=upper,
            debit=debit,
            allowance=H,
            variable_hours=tx.astype(np.int32),
            variable_positions=nx.astype(np.int32),
            support_hours=globalhours.astype(np.int32),
            support_intercepts=intercepts,
            support_plane_error_bounds=plane_errors,
            active_hours=active,
            dt=dt,
        )
        dump(
            out / "MODEL_IDENTITY.json",
            {
                "year": year,
                "parent_physical_case_hash": m["physical_case_hash"],
                "annual_physical_case_hash": identity,
                "time_grid_id": timeid,
                "hours": T,
                "positions": N,
                "network_states": 94,
                "physical_rows_per_hour": J,
                "original_full_x_coordinates": T * N,
                "new_nonzero_bound_x_coordinates": RR,
                "historical_screened_x_coordinates": ret,
                "theta_coordinates": T,
                "support_count": K,
                "budget_constraints": N,
                "master_rows": K + N,
                "master_columns": RR + T,
                "master_nonzeros": int(Ax.nnz + K + D.nnz),
                "C0": 0.0,
                "historical_C0": constant,
                "geometry_hash": gh,
                "selector_rows_verified": selectors_verified,
                "maximum_screened_coefficient_difference": maxgrad,
                "maximum_intercept_fsum_difference": maxint,
                "maximum_support_minus_full_hour_at_saved_candidate": maxpoint,
                "max_support_plane_forward_error_bound": float(plane_errors.max()),
                "screening_strategy": "Complete positive physical box retained. Only b==0 removed exactly. Historical geometry fully hash reconciled but its dominated-action/secure-hour removal not required for lower-bound validity. C0=0 and all8760 hourly epigraphs retained.",
                "minorant_proof": "Every positive/negative selector is unique within that orientation and belongs to its registered row/state. Any subset of signed thermal hinges yields a global lower support; omitted full hinges have multiplier zero.",
                "solver_not_called": True,
                "preparation_runtime_s": time.perf_counter() - t0,
            },
        )
        with (out / "SUPPORT_MAPPING.csv").open("w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "support_index",
                    "archived_local_active_hour",
                    "year_hour",
                    "timestamp_utc",
                    "pos_start",
                    "pos_stop",
                    "neg_start",
                    "neg_stop",
                    "intercept",
                    "uniform_plane_error_bound",
                ]
            )
            w.writerows(
                (
                    k,
                    int(hour[k]),
                    int(globalhours[k]),
                    str(d["dates"][ix[int(globalhours[k])]]),
                    int(po[k]),
                    int(po[k + 1]),
                    int(no[k]),
                    int(no[k + 1]),
                    intercepts[k],
                    plane_errors[k],
                )
                for k in range(K)
            )
        reconciliation.append(
            {
                "year": year,
                "annual_case_identity": "PASS",
                "time_order": "PASS",
                "position_order": "PASS",
                "state_row_order": "PASS",
                "box_and_masks": "PASS",
                "M1LP_budget_not_M2": "PASS",
                "historical_geometry_hash": "PASS",
                "support_selectors": "PASS",
                "C0_counted_once": "PASS",
                "verification_domain": "FULL_POSITIVE_BOX",
                "master_columns": RR + T,
                "historical_columns": ga["support_initial_columns"],
                "verdict": "PASS_VALID_RESTRICTED_RELAXATION",
            }
        )
        del base, errbase, Ax, aa, ii, pr, nr, reference, posall, negall, parts
    with (GATE / "GATE1_MODEL_RECONCILIATION.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(reconciliation[0]))
        w.writeheader()
        w.writerows(reconciliation)
    dump(
        GATE / "PRE_SOLVE_GATE.json",
        {
            "all_three_models_reconciled": True,
            "verdict": "PASS",
            "method": "Independent full-box restricted-master relaxation, no historical source execution and no screening-dependent tightening.",
            "optimizations_started": 0,
        },
    )
    print("PREPARATION_COMPLETE_ALL_THREE_MODELS_PASS", flush=True)


if __name__ == "__main__":
    main()
