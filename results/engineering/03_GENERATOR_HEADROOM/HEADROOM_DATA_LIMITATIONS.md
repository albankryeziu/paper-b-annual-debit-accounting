# Generator headroom: identification limits

The final A identifies participant and balanced counter-injection bus components, not generator-specific adjustments. Its exact decomposition is A[:,n]=sigma_n(C_P[:,n]-pi). Thus p(x)=p_background-C_P diag(sigma)(v-x)+pi sum_n sigma_n(v_n-x_n). No net bus injection is treated as generation.

Native generator and interchange Pmin/Pmax and Qmin/Qmax bounds, source buses, reference PV dispatch and security-adjusted background device dispatch are recoverable. The AC evaluator fixes PV P and places all remaining net P in synthetic PQ devices; slack P covers losses. It does not impose generator active-power feasibility on those synthetic injections.

Generator-specific participation factors for A are NOT IDENTIFIABLE FROM AVAILABLE DATA. Policy-dependent generator headroom violations, counts of affected generators, reserve availability, dispatch availability and ramp feasibility therefore cannot be asserted. Native Pmax is a model bound, not independently verified installed/available capacity. Simple Q bounds are not complete Q-V capability curves.

The bus-aggregate table is an explicitly conditional static necessary-condition audit. It separates recovered background generator/boundary dispatch from load/renewable curtailment, then asks whether assigning the entire balanced counter-injection to that same native generator/interchange fleet would exceed aggregate bounds, holding all other components fixed. It does not assign individual devices, change generation or repair policies. It is neither a proof of general physical infeasibility nor a proof of deployability; alternative load/renewable contributions are outside that conditional test. Background boundary totals are frozen in the DC builder, so allowing their native device ranges in this diagnostic is already a relaxation of that construction.

All scheduled policy injections remain the same across retained line/transformer contingencies. No post-contingency generator redispatch, generator outage, frequency control, reserve or ramp model is introduced. Full-year bounds are evaluated on the one-hour grid only. No ramp values are inferred from capacity.
