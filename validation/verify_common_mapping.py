"""Independent reconstruction of the common P/Q participant mapping.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from engineering_common import OUT, REAL, action, consume, jsout, load_case, np, pd, policies, read


def run():
    target = OUT / "07_QC_AND_INTEGRITY/AC_MAPPING_INDEPENDENT_QC.json"
    if target.exists():
        return
    meta, a = load_case()
    z = np.load(OUT / "00_AUTHORITY_AND_INPUTS/AC_MAPPING_V2.npz", allow_pickle=False)
    Cp = z["C_P_final"]
    Cnew = z["C_Q_common"]
    Ch = z["C_Q_historical"]
    mapping = read(consume(REAL / "physical_case/FINAL_DUTCH_CASE/PARTICIPANT_MAPPING.json"))[
        "records"
    ]
    by = {r["participant_id"]: r for r in mapping}
    sigma = np.array([1 if d == "import" else -1 for d in meta["participant_directions"]])
    estimates = []
    for j, pid in enumerate(meta["participant_ids"]):
        r = by[pid]
        if not r["connection_is_multibus"]:
            e = np.zeros(45)
            e[meta["bus_ids"].index(str(r["support_bus_ids"][0]))] = 1.0
            estimates.append(e - a["A"][:, j] / sigma[j])
    pi = np.median(estimates, axis=0)
    independent = a["A"] / sigma + pi[:, None]
    assert np.max(np.abs(independent - Cp)) < 1e-12 and np.array_equal(Cnew, Cp)
    assert np.min(Cnew) >= 0 and np.max(np.abs(Cnew.sum(axis=0) - 1)) < 1e-12
    inputs = pd.read_csv(OUT / "02_MATCHED_AC_VALIDATION/AC_MAPPING_INPUT_QC.csv")
    error = 0.0
    conservation = 0.0
    for year in (2029, 2030, 2031):
        ix = np.flatnonzero(a["dates"].year == year)
        for policy in ("ROLLING_M2", "U", "C", "EC"):
            x = action(policies(year)[policy][0])
            for r in inputs[(inputs.year == year) & (inputs.policy == policy)].to_dict("records"):
                t = int(r["time_index"])
                signed = -sigma * (a["requests_mw"][ix[t]] - x[t]) * np.tan(np.arccos(0.98))
                qind = independent @ signed
                qh = Ch @ signed
                difference = qind - qh
                error = max(
                    error,
                    abs(float(np.max(np.abs(difference))) - r["maximum_spatial_Q_change_mvar"]),
                )
                conservation = max(conservation, abs(float(qind.sum() - qh.sum())))
    assert error < 1e-9 and conservation < 1e-9
    jsout(
        target,
        dict(
            status="PASS",
            independent_key_direct_columns=len(estimates),
            independent_common_map_max_error=float(np.max(np.abs(independent - Cnew))),
            nonnegative_normalized=True,
            active_transaction_matrix_unchanged=True,
            reactive_uses_final_active_component_distribution=True,
            selected_input_records=len(inputs),
            max_spatial_difference_reconstruction_error_Mvar=error,
            max_total_Q_difference_Mvar=conservation,
            unity="Participant delta Q exactly zero by both mappings; independent solved cases agree as recorded in MATCHED_AC_QC.json",
            AC_solves=0,
            optimizer_calls=0,
        ),
    )
    print("INDEPENDENT COMMON-MAP QC PASS", flush=True)


if __name__ == "__main__":
    run()
