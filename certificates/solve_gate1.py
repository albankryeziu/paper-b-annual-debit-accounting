"""Final numerical certificate reconstruction and verification.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

import sys

sys.dont_write_bytecode = True
from gate1_common import *
import scipy.sparse as sp
import gurobipy as gp
import subprocess, time, os, gzip, shutil


def attr(model, name):
    try:
        value = getattr(model, name)
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    except (AttributeError, gp.GurobiError):
        return None


def main():
    assert read(GATE / "PRE_SOLVE_GATE.json")["all_three_models_reconciled"]
    for year in [2029, 2030, 2031]:
        folder = GATE / f"year_{year}"
        ident = read(folder / "MODEL_IDENTITY.json")
        z = np.load(folder / "MODEL_ARRAYS.npz", allow_pickle=False)
        Ax = sp.load_npz(folder / "A_support.npz")
        D = sp.load_npz(folder / "D_budget.npz")
        upper = z["upper"]
        H = z["allowance"]
        hh = z["support_hours"]
        intercept = z["support_intercepts"]
        RR = len(upper)
        T = len(z["dt"])
        K = len(hh)
        marker = folder / "OPTIMIZATION_STARTED.json"
        with marker.open("x", encoding="utf8") as f:
            json.dump(
                {
                    "year": year,
                    "annual_problem_id": f"INDEPENDENT_M1LP_{year}",
                    "only_authorized_problem": True,
                    "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "code_sha256": sha(__file__),
                },
                f,
                indent=2,
            )
        model = gp.Model(f"INDEPENDENT_M1LP_{year}")
        model.Params.LogFile = str(folder / "GATE1_SOLVER.log")
        model.Params.LogToConsole = 1
        model.Params.OutputFlag = 1
        model.Params.Threads = 1
        model.Params.Method = 1
        model.Params.Presolve = 2
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.NumericFocus = 2
        model.Params.Seed = 0
        model.Params.TimeLimit = 3600
        allvars = model.addMVar(
            RR + T,
            lb=np.zeros(RR + T),
            ub=np.concatenate([upper, np.full(T, gp.GRB.INFINITY)]),
            name="physical_x_then_hour_theta",
        )
        budgets = model.addMConstr(
            sp.hstack([D, sp.csr_matrix((114, T))], format="csr"),
            allvars,
            "<",
            H,
            name="M1LP_request_normalized_budget",
        )
        theta = sp.csr_matrix((-np.ones(K), (np.arange(K), hh)), shape=(K, T))
        cons = model.addMConstr(
            sp.hstack([Ax, theta], format="csr"),
            allvars,
            "<",
            -intercept,
            name="archived_valid_hourly_support",
        )
        model.setObjective(np.concatenate([np.zeros(RR), np.ones(T)]) @ allvars, gp.GRB.MINIMIZE)
        model.update()
        assert (
            model.NumVars == ident["master_columns"]
            and model.NumConstrs == ident["master_rows"]
            and model.NumNZs == ident["master_nonzeros"]
        )
        # Windows Gurobi delegates compressed export to an external compression
        # utility. Export plain MPS, then compress with Python after disposal.
        model.write(str(folder / "RESTRICTED_MASTER.mps"))
        dump(
            folder / "SOLVER_SETTINGS.json",
            {
                "solver": "Gurobi",
                "version": list(gp.gurobi.version()),
                "Threads": 1,
                "Method": 1,
                "Presolve": 2,
                "FeasibilityTol": 1e-9,
                "OptimalityTol": 1e-9,
                "NumericFocus": 2,
                "Seed": 0,
                "TimeLimit_s": 3600,
                "objective_constant": 0,
                "matrix_scaling": "Gurobi default ScaleFlag; no manual variable/objective rescaling",
                "differences_from_historical": "Presolve enabled and tighter feasibility/optimality tolerances for this separate independent master; original physical/accounting coefficients unchanged.",
            },
        )
        print("AUTHORIZED_ANNUAL_SOLVE_START", year, RR + T, K + 114, flush=True)
        started = time.perf_counter()
        model.optimize()
        result = {
            "year": year,
            "solver_status_code": model.Status,
            "solver_status": "OPTIMAL" if model.Status == gp.GRB.OPTIMAL else str(model.Status),
            "objective": attr(model, "ObjVal"),
            "solver_bound": attr(model, "ObjBound"),
            "runtime_s": attr(model, "Runtime"),
            "elapsed_s": time.perf_counter() - started,
            "iterations": attr(model, "IterCount"),
            "variables": model.NumVars,
            "constraints": model.NumConstrs,
            "nonzeros": model.NumNZs,
            "ConstrVio": attr(model, "ConstrVio"),
            "BoundVio": attr(model, "BoundVio"),
            "DualVio": attr(model, "DualVio"),
            "ComplVio": attr(model, "ComplVio"),
            "Kappa": attr(model, "Kappa"),
            "KappaExact": attr(model, "KappaExact"),
            "Gurobi_MaxMemUsed_GB": attr(model, "MaxMemUsed"),
            "support_count_initial": K,
            "supports_added": 0,
            "annual_problems_solved_so_far": year - 2028,
            "StageII_solved": False,
        }
        dump(folder / "SOLVER_RESULT.json", result)
        if model.SolCount == 0:
            raise RuntimeError("NO_PRIMAL_SOLUTION; no further model authorized automatically")
        try:
            np.savez_compressed(
                folder / "FULL_SOLVER_VECTORS.npz",
                x=allvars.X[:RR],
                theta=allvars.X[RR:],
                budget_pi=budgets.Pi,
                support_pi=cons.Pi,
                reduced_cost=allvars.RC,
                budget_slack=budgets.Slack,
                support_slack=cons.Slack,
                variable_basis=allvars.VBasis,
                budget_basis=budgets.CBasis,
                support_basis=cons.CBasis,
            )
        except gp.GurobiError as e:
            raise RuntimeError("Full independent dual vectors unavailable: " + str(e))
        model.write(str(folder / "RESTRICTED_MASTER.bas"))
        model.dispose()
        del Ax, D, model, allvars
        with (
            (folder / "RESTRICTED_MASTER.mps").open("rb") as src,
            gzip.open(folder / "RESTRICTED_MASTER.mps.gz", "wb", compresslevel=3) as dst,
        ):
            shutil.copyfileobj(src, dst, 1048576)
        (folder / "RESTRICTED_MASTER.mps").unlink()
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with (folder / "INDEPENDENT_CERTIFICATE_VERIFIER.log").open("w", encoding="utf8") as f:
            done = subprocess.run(
                [sys.executable, str(GATE / "verify_certificate.py"), str(year)],
                cwd=str(GATE),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if done.returncode:
            raise RuntimeError(
                "Independent certificate verifier failed; stop before further annual models"
            )
        cert = read(folder / "INDEPENDENT_CERTIFICATE.json")
        if cert["verdict"] == "FAIL":
            raise RuntimeError("Scientific certificate verification failed; Gate3 prohibited")
    dump(
        GATE / "ALL_SOLVES_COMPLETE.json",
        {
            "annual_problems": 3,
            "years": [2029, 2030, 2031],
            "extra_families": 0,
            "StageII": 0,
            "full_campaign_runs": 0,
            "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    print("ALL_THREE_AUTHORIZED_ANNUAL_PROBLEMS_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        dump(
            GATE / "GATE1_STOP.json",
            {
                "reason": repr(exc),
                "scientific_preparation_stopped": True,
                "checkpoints_preserved": True,
                "utc": pd.Timestamp.now(tz="UTC").isoformat(),
            },
        )
        raise
