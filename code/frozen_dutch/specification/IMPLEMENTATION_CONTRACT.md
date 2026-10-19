# Independently reconstructed implementation contract

Authority: supplied final manuscript, then supplied technical report. This
contract precedes changes to the supplied compact implementation.

For each participant n and hour t, P>0, explicit contract mask kappa and
eligibility chi<=kappa define b=chi*min(P,request). The balanced injection is
p_bg-A*request+A*x; each state flow is H*p+h with its own positive rating F.
All policies choose 0<=x<=b. Physical congestion is
S(x)=sum(dt*max(0,f-F,-f-F)); E(x)=sum(dt*x).
Stage I minimizes S. Stage II minimizes E subject to S<=S_star+epsilon_S,
where epsilon_S=1e-6+1e-9*max(1,abs(S_star)). No production selector is needed.
The optional reporting selector must retain both scientific faces and cannot
supply an M3 teacher. Numerical comparison envelopes do not change these faces.

* M1 adds binary u, min(b,max(1,0.01P))*u<=x<=b*u and sum(dt*u)<=H.
* M1-LP eliminates relaxed u by u=x/b on b>0. This satisfies the lower link
  because the materiality floor is at most b; debit is sum(dt*x/b)<=H.
* M2 debit is sum(dt*x/P)<=H. Since b<=P it contains M1-LP.
* M3 debit is sum(dt*q*x/P)<=H for a positive frozen card q.
* M2-MATCHED uses exactly Hhat=sum(dt*x_M3/P), participant by participant,
  with the completed source solution hash. The source trajectory is feasible.
* Annual models know the complete scenario year and are benchmarks.
* Rolling models see only the 48-hour window and remaining allowance; they
  implement 24 hours, debit only those hours, and truncate the final window.
  No reserve, terminal value, physical intertemporal device state or pacing
  enters the parent problem. Restart recomputes state from accepted past blocks.

Annual continuous optimization uses one hourly epigraph theta and generated
supports of the sum of directed hinges. Selecting any subset of hinges gives
an affine minorant; selecting positive hinges makes it tight at the candidate.
The master is a lower relaxation, so full oracle and physical replay closure
are necessary. Secure zero-action hours and actions unable to decrease any
potential hinge are dominated by zero because energy and debit are nonnegative.
Fixed-zero compression is the exact substitution x=P_active*x_active. Cached
selectors are hints only; coefficients are rebuilt from immutable physical data.

M2 teacher mu=-Pi_budget has nonnegative scarcity sign and is paired with the
primal from the same Stage-II LP. Alpha=mu*dt*x/P. Fit 2025--2027 by direction;
validate complete 2028 operator outcomes. F0 is calendar, capacity, request
fraction, zone and technology only. Select by S, E, six-rate standard deviation,
ridge before histogram boosting, then stable candidate ID. Freeze before any
2029--2031 feature values or physical outcomes are opened by fitting code.
Input preparation may partition immutable data and compute contractual allowance
metadata; fitting receives only the 2025--2028 partition and that metadata.

The rate QP minimizes A*q+0.5*lambda*sum(W*(q-1)^2), lambda=0.001 MW,
ordered descending rates, primary bounds [0.5,1.5], sum(W*q)=sum(W).
Sparse directions are flat. Controls use PTDF net relief, inverse ordering,
and 32 within-direction shuffles with seeds 118001..118032. Their amplitude
projection uses the selected card and their own fitting reference exposure.
Gamma uses the supplied 21-point grid, first screening 0.7,1,1.3. Endpoints
0 and 1 are exact objects; other rates solve the declared weighted projection
under [0.25,2]. Rate-bound cases are [0.7,1.3], [0.5,1.5], [0.25,2].

Price, matched allowances, independent DC and selected AC are ex post only.
Q freezes selection before DC/AC results; AC failures remain evidence and do
not modify the DC policy. Stage 07 produces evidence, never numerical prose.

Specification inconsistencies requiring explicit interpretation: the manuscript's
suggestion of positive M3 improvement over optimized M2-MATCHED is incompatible
with its own containment theorem. Report exposure effect and nonnegative shape
penalty, never claim such an improvement. With nonzero epsilon_S, set inclusion
alone does not imply ordering of E on different approximate Stage-I faces;
the strict Stage-I certificate remains valid. The main manuscript takes priority
over older configuration fields for engineering selection and reduction gates.


The direct-position reduction benchmark supplies the 95th-percentile worst-state
relative error thresholds and 5th-percentile agreement thresholds. No legacy
fixed error thresholds substitute for this rule. On the supplied sources all 35
multibus positions pass, selecting all 114 positions. Input stage 01 reconstructs
this gate before fitting and creates a new physical-case identity. It retains
hash-verified background arrays and the frozen network only after raw-source
reconstruction passes the existing 1e-7 MW physical-identity error envelope.
This is input construction, not re-labeling an old optimized policy.
