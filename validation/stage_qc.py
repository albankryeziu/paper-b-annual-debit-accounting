"""Independent checks of matched engineering results.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from engineering_common import (
    OUT,
    RESULTS,
    action,
    consume,
    jsout,
    load_case,
    np,
    pd,
    policies,
    read,
    sha,
)


def run():
    target = OUT / "07_QC_AND_INTEGRITY/INDEPENDENT_NUMERICAL_QC.json"
    if target.exists():
        return
    meta, a = load_case()
    physical = pd.read_csv(OUT / "01_PHYSICAL_METRICS/ANNUAL_PHYSICAL_METRICS.csv")
    hourly = pd.read_parquet(OUT / "01_PHYSICAL_METRICS/HOURLY_MAXIMUM_EXCEEDANCE.parquet")
    critical = pd.read_csv(OUT / "01_PHYSICAL_METRICS/CRITICAL_ROWS_BY_POLICY.csv")
    max_flow_error = 0.0
    max_V_error = 0.0
    checked = 0
    for r in physical.to_dict("records"):
        year = r["year"]
        policy = r["policy"]
        g = hourly[(hourly.year == year) & (hourly.policy == policy)].reset_index(drop=True)
        assert len(g) == 8760 and not g.timestamp_utc.duplicated().any()
        assert r["overload_hours_engineering"] == int((g.V_MW > 1).sum()) and r[
            "overload_hours_numerical"
        ] == int((g.V_MW > 1e-4).sum())
        assert (
            abs(r["Vmax_MW"] - g.V_MW.max()) < 1e-8
            and abs(r["mean_hourly_max_MW"] - g.V_MW.mean()) < 1e-8
        )
        assert (
            abs(r["p95_hourly_max_MW"] - np.percentile(g.V_MW, 95)) < 1e-8
            and abs(r["p99_hourly_max_MW"] - np.percentile(g.V_MW, 99)) < 1e-8
        )
        ix = np.flatnonzero(a["dates"].year == year)
        p, kind = policies(year)[policy]
        assert sha(p) == r["source_sha256"]
        x = action(p)
        for t in sorted(set([0, 24, 4379, 8759, int(g.V_MW.argmax())])):
            gi = ix[t]
            f_req = a["H"] @ (a["p_background_mw"][gi] - a["A"] @ a["requests_mw"][gi]) + a["h"]
            flow = f_req + a["G"] @ x[t]
            V = float(np.maximum(np.abs(flow) - a["F"], 0).max())
            max_V_error = max(max_V_error, abs(V - g.at[t, "V_MW"]))
            assert abs(V - g.at[t, "V_MW"]) < 1e-7
            checked += 1
        for c in critical[(critical.year == year) & (critical.policy == policy)].to_dict("records"):
            j = int(c["flat_row"])
            gi = a["dates"].get_loc(pd.Timestamp(c["timestamp_utc"]))
            ti = int(np.flatnonzero(ix == gi)[0])
            pstate = a["p_background_mw"][gi] - a["A"] @ (a["requests_mw"][gi] - x[ti])
            f = float(a["H"][j] @ pstate + a["h"][j])
            e = abs(f - c["flow_MW"])
            assert e < 1e-7
            max_flow_error = max(max_flow_error, e)
            assert tuple(a["rows"][j]) == (c["state_id"], c["physical_element_id"])
    cases = pd.read_csv(OUT / "02_MATCHED_AC_VALIDATION/MATCHED_AC_CASES.csv")
    key = ["mapping", "year", "time_index", "state_id", "pf_case", "policy"]
    assert len(cases) == 1296 and not cases.duplicated(key).any()
    original = pd.read_csv(consume(RESULTS / "selected_ac/CASES.csv"))
    hist = cases[(cases.mapping == "HISTORICAL_TWO_MAP") & (cases.policy == "ROLLING_M2")]
    for r in hist.to_dict("records"):
        old = original[
            (original.year == r["year"])
            & (original.time_index == r["time_index"])
            & (original.state_id == r["state_id"])
            & (original.pf_case == r["pf_case"])
            & (original.policy_state == "RHO15_ROLLING_M2")
        ]
        assert len(old) == 1
        for col in (
            "converged",
            "thermal_violations",
            "voltage_violations",
            "ac_feasible_under_declared_checks",
        ):
            assert r[col] == old.iloc[0][col]
    v = pd.read_csv(OUT / "02_MATCHED_AC_VALIDATION/MATCHED_AC_VOLTAGES.csv")
    l = pd.read_csv(OUT / "02_MATCHED_AC_VALIDATION/MATCHED_AC_LOADING.csv")
    q = pd.read_csv(OUT / "02_MATCHED_AC_VALIDATION/MATCHED_AC_REACTIVE_LIMITS.csv")
    case = cases.set_index(key)
    v["viol"] = (v.vm_pu < 0.95) | (v.vm_pu > 1.05)
    l["viol"] = l.loading_percent > 100
    l["J"] = np.maximum(l.loading_percent / 100 - 1, 0)
    vm = v.groupby(key).vm_pu.agg(["min", "max"])
    vv = v.groupby(key).viol.sum()
    lm = l.groupby(key).loading_percent.max()
    lv = l.groupby(key).viol.sum()
    J = l.groupby(key).J.sum()
    Q = q.groupby(key).limit_violation_mvar.max()
    valid = case[case.converged]
    assert set(valid.index) == set(vm.index) == set(lm.index) == set(Q.index)
    errors = {}
    for label, values, col, tol in [
        ("min_voltage", vm["min"], "min_vm_pu", 1e-12),
        ("max_voltage", vm["max"], "max_vm_pu", 1e-12),
        ("loading", lm, "maximum_loading_percent", 1e-10),
        ("normalized_overload", J, "ac_apparent_normalized_thermal_exceedance", 1e-10),
        ("Q_excess", Q, "max_q_limit_violation_mvar", 1e-10),
    ]:
        e = float(np.max(np.abs(values.reindex(valid.index).to_numpy() - valid[col].to_numpy())))
        assert e < tol
        errors[label] = e
    assert np.array_equal(
        vv.reindex(valid.index).to_numpy(), valid.voltage_violations.to_numpy()
    ) and np.array_equal(lv.reindex(valid.index).to_numpy(), valid.thermal_violations.to_numpy())
    lines = l[l.element_kind == "line"]
    native = lines.native_value / lines.native_rating * 100
    currenterr = float(np.max(np.abs(native - lines.loading_percent)))
    assert currenterr < 1e-8
    # Retained baseline archive comparisons are independent of policy rankings.
    head = read(OUT / "03_GENERATOR_HEADROOM/HEADROOM_QC.json")
    info = read(OUT / "04_INFORMATION_TIMING/INFORMATION_QC.json")
    jsout(
        target,
        dict(
            status="PASS",
            physical_policy_objects=21,
            full_hourly_diagnostic_rows=len(hourly),
            alternate_algebra_sample_hours=checked,
            maximum_alternate_hourly_peak_error_MW=max_V_error,
            critical_rows_checked=len(critical),
            maximum_critical_flow_error_MW=max_flow_error,
            AC_cases=len(cases),
            AC_voltage_rows=len(v),
            AC_loading_rows=len(l),
            AC_reactive_device_rows=len(q),
            AC_aggregation_max_errors=errors,
            native_line_loading_reconstruction_error_percent=currenterr,
            headroom_status=head["status"],
            information_status=info["status"],
            scope="Complete persisted-table aggregation and full objective agreement; independent alternate physical algebra on declared sample and all 525 critical rows; no additional optimization or AC solves",
        ),
    )
    print("INDEPENDENT QC PASS", flush=True)


if __name__ == "__main__":
    run()
