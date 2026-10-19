# Information timing and the intended 07:00 implementation

The executed rolling experiment has a deterministic 48-hour look-ahead and commits the first 24 hours at each epoch. At epoch start s, the optimizer receives physical-case rows s:min(s+48,T), including requests v, eligibility and action bounds b, background bus injections, durations and the corresponding debit-rate rows. Network matrices, ratings, participant capacities, fixed balancing key and contract allowances are fixed. The remaining annual debit is carried from already committed actions. The last window is 24 hours. This is finite-window deterministic look-ahead, distinct from the full-year annual optimization.

The inputs are realized controlled-scenario values. They are not authenticated forecast vintages issued at the historical decision time. The experiments do not enforce 07:00 announcements, forecast revisions, quarter-hour decision intervals, a market gate closure, or measurability of the schedule with respect to a timestamped forecast information set. No claim of real-time deployment or regulatory execution is established.

The conservative card uses the requested-state right derivative, m_nt=-P_n psi'_t(0;e_n). The function `hinge_relief_scores` evaluates f_requested=H(p_background-A v)+h, sums signed marginal relief over all monitored rows, includes exact boundary directional hinge terms, and sets scores to zero where the action bound is zero. Neither a future optimized x nor a future annual dual variable enters this score. Nevertheless future-hour scenario requests/backgrounds are known to this retrospective calculation. An exogenous score is not automatically an authentic contemporaneous forecast.

Fitting uses 2025-2027; 2028 is the promotion/validation year. The stored card, bucket thresholds, rate levels and uniform reference premium remain fixed across 2029-2031. Actual q_nt values differ with the scenario requested-state bucket; the fixed card does not imply an identical time-by-position rate matrix in different years. All six held-out experiment records retain the same card hash, 48-hour horizon and 24-hour implementation period. This audit does not refit or retrain any parameter.

The annual M1-LP lower bounds and feasible M2 witnesses concern the frozen full-year DC model. They do not depend on the new reactive mapping, nor do they bound the performance of every prospective rolling contract.

## Prospective specification, not a validated controller

Let F_tau be the information available at announcement time tau, and let hats denote forecasts conditional on that information. Construct forecast requested injections p_hat_req,t|tau=p_hat_background,t|tau-A v_hat_t|tau and the corresponding forecast hinge function psi_hat_t|tau. Then

`m_hat_nt|tau = -P_n * psi_hat'_t|tau(0;e_n)`.

Apply the frozen bucket thresholds/rates to that score and require both the announced debit rates and the committed restriction schedule to be F_tau-measurable. State explicitly whether announced rates can be revised or must remain binding; define how revisions, changed eligibility, measurement errors, incomplete realization, debit settlement and annual carried balances are handled. The objective and controller information set must match those announcement rules.

Testing this requires archived issue-time-stamped forecasts and revisions for participant requests, load, renewable generation and background operating conditions, plus network/topology availability at issue time, contract eligibility, the applicable timezone/DST rule and interval resolution, announcement/commitment gates and meter-settlement definitions. The intended 07:00 timestamp must be tied to those concrete forecast products and contractual rules. None is manufactured here. No AR fit, synthetic forecast noise, rolling optimization or forecasting experiment is performed.

## Why finite deterministic look-ahead is not a universal value bound

Perfect information within a 48-hour window does not make a receding-horizon policy globally optimal or dominate every prospective heuristic. For example, with a single annual unit of debit, a unit of relief at the first hour and two units of relief at hour 49, a 48-hour controller can spend its budget on the visible first-hour benefit. A feasible controller that reserves the budget obtains the later benefit. The finite-window controller has perfect information on every window it sees, but earns less total relief. A global optimal-value information bound would require common action domains, a common full decision horizon, the relevant nonanticipativity constraints, and exact optimization under comparable information. No such upper-bound claim is made for these rolling trajectories.

## Verification scope

The source slices and six experiment/card records are hash-bound in the consumed-source inventory. The earlier independent rate-assignment audit is retained as original evidence; this stage verifies the current record identities, rate levels and horizon settings. A fresh authenticated forecast-vintage audit is impossible without those forecast products and is not claimed.
