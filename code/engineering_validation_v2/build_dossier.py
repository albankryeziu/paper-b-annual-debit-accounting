"""Evidence-led figures, engineering reports and proposed amendments; no solves."""
from engineering_common import *
import subprocess,io

POLICIES=['ANNUAL_M1_LP','ANNUAL_M2','ROLLING_M1','ROLLING_M2','U','C','EC']
LABELS={'ANNUAL_M1_LP':'Annual M1-LP','ANNUAL_M2':'Annual M2','ROLLING_M1':'Rolling M1','ROLLING_M2':'Rolling M2','U':'U','C':'C','EC':'EC witness'}

def table(frame):
 frame=frame.copy()
 for col in frame:
  if pd.api.types.is_float_dtype(frame[col]):frame[col]=frame[col].map(lambda x:f'{x:,.3f}')
 return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))+'\n\n'

def figures(physical,cases,summary):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
 folder=OUT/'06_REPORTS/figures';folder.mkdir(exist_ok=True)
 years=[2029,2030,2031];colors=dict(zip(POLICIES,plt.get_cmap('tab10').colors[:7]))
 def save(fig,name):
  fig.tight_layout()
  if name=='FIGURE_A_RESIDUAL_AND_PEAK':fig.subplots_adjust(bottom=.34)
  for ext in ('pdf','png'):
   buf=io.BytesIO();fig.savefig(buf,format=ext,bbox_inches='tight');atomic(folder/f'{name}.{ext}',buf.getvalue())
  plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(7.1,3.1),sharey=True)
 for ax,y in zip(axes,years):
  for p in POLICIES:
   r=physical[(physical.year==y)&(physical.policy==p)].iloc[0];ax.scatter(r.S_mwh_row_state/1e6,r.Vmax_MW,s=25,color=colors[p],label=LABELS[p])
  ax.set(title=str(y),xlabel='Integrated residual\n(million MW h row-state)',xlim=(0,None),ylim=(0,2500));ax.grid(alpha=.2)
 axes[0].set_ylabel('Maximum exceedance (MW)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,0));save(fig,'FIGURE_A_RESIDUAL_AND_PEAK')
 fig,axes=plt.subplots(1,3,figsize=(7.1,2.5),sharey=True)
 for ax,y in zip(axes,years):
  g=physical[physical.year==y].set_index('policy').loc[POLICIES];ax.bar(range(7),g.overload_hours_engineering,color=[colors[p] for p in POLICIES]);ax.set_xticks(range(7));ax.set_xticklabels(['A-LP','A-M2','R-M1','R-M2','U','C','EC'],rotation=45);ax.set(title=str(y),ylim=(0,8760));ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Hours with any exceedance >1 MW');save(fig,'FIGURE_B_MATERIAL_OVERLOAD_HOURS')
 hourly=pd.read_parquet(OUT/'01_PHYSICAL_METRICS/HOURLY_MAXIMUM_EXCEEDANCE.parquet');fig,axes=plt.subplots(1,3,figsize=(7.1,2.55),sharey=True)
 for ax,y in zip(axes,years):
  for p in ('ROLLING_M2','U','C','EC'):
   v=np.sort(hourly[(hourly.year==y)&(hourly.policy==p)].V_MW.to_numpy())[::-1];ax.step(np.arange(1,len(v)+1),v,where='post',label=LABELS[p],color=colors[p],linewidth=1)
  ax.set(title=str(y),xlabel='Hours in exceedance-duration order',xlim=(0,8760),ylim=(0,None));ax.grid(alpha=.2)
 axes[0].set_ylabel('Hourly maximum exceedance (MW)');axes[-1].legend();save(fig,'FIGURE_C_HOURLY_EXCEEDANCE_DISTRIBUTION')
 fig,axes=plt.subplots(1,2,figsize=(7.1,2.6),sharey=True)
 for ax,mapping,title in zip(axes,['HISTORICAL_TWO_MAP','COMMON_PQ_V2'],['Historical two-map sensitivity','Common participant map v2']):
  for j,p in enumerate(('ROLLING_M2','U','C','EC')):
   g=summary[(summary.mapping==mapping)&(summary.policy==p)].set_index('year').loc[years];ax.bar(np.arange(3)+(j-1.5)*.18,g.J_AC_sum,width=.18,label=LABELS[p],color=colors[p])
  ax.set_xticks(range(3));ax.set_xticklabels(years);ax.set_title(title);ax.set_ylim(0,None);ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Selected-case sum\nof normalized AC overload');axes[1].legend();save(fig,'FIGURE_D_MATCHED_AC_OVERLOAD')
 fig,axes=plt.subplots(1,2,figsize=(7.1,2.6));pp=['ROLLING_M2','U','C','EC'];xx=np.arange(4)
 for shift,mapping,color in [(-.18,'HISTORICAL_TWO_MAP','#667d8e'),(.18,'COMMON_PQ_V2','#2a7c7a')]:
  g=cases[cases.mapping==mapping].groupby('policy');n=g.apply(lambda f:int(((f.thermal_violations==0)&f.converged).sum())).reindex(pp);peak=g.maximum_loading_percent.max().reindex(pp)
  axes[0].bar(xx+shift,n,width=.34,label='Historical' if shift<0 else 'Common v2',color=color);axes[1].bar(xx+shift,peak,width=.34,color=color)
 for ax in axes:ax.set_xticks(xx);ax.set_xticklabels(['M2','U','C','EC']);ax.set_ylim(0,None);ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Thermal-pass cases\n(of 162 per policy/map)');axes[0].set_ylim(0,162);axes[0].legend();axes[1].set_ylabel('Maximum native-current loading (%)');axes[1].axhline(100,color='black',linewidth=.8,linestyle='--');save(fig,'FIGURE_E_AC_THERMAL_FEASIBILITY')
 jsout(folder/'FIGURE_DATA_PROVENANCE.json',dict(A_B='01_PHYSICAL_METRICS/ANNUAL_PHYSICAL_METRICS.csv',C='01_PHYSICAL_METRICS/HOURLY_MAXIMUM_EXCEEDANCE.parquet',D='02_MATCHED_AC_VALIDATION/MATCHED_AC_SUMMARY.csv',E='02_MATCHED_AC_VALIDATION/MATCHED_AC_CASES.csv',smoothing=False,statistical_error_bars=False,years_are_independent_trials=False))

def run():
 target=OUT/'06_REPORTS/DOSSIER_STATUS.json'
 if target.exists():return
 require_preflight()
 from verify_common_mapping import run as verify_mapping
 verify_mapping()
 from stage_qc import run as qc
 qc()
 physical=pd.read_csv(OUT/'01_PHYSICAL_METRICS/ANNUAL_PHYSICAL_METRICS.csv');cases=pd.read_csv(OUT/'02_MATCHED_AC_VALIDATION/MATCHED_AC_CASES.csv');summary=pd.read_csv(OUT/'02_MATCHED_AC_VALIDATION/MATCHED_AC_SUMMARY.csv');pairs=pd.read_csv(OUT/'02_MATCHED_AC_VALIDATION/MATCHED_AC_PAIRWISE_SUMMARY.csv');sensitivity=pd.read_csv(OUT/'02_MATCHED_AC_VALIDATION/AC_MAPPING_SENSITIVITY.csv');head=pd.read_csv(OUT/'03_GENERATOR_HEADROOM/BALANCING_FEASIBILITY_BY_POLICY.csv');acqc=read(OUT/'02_MATCHED_AC_VALIDATION/MATCHED_AC_QC.json')
 # PDF skill operation marker: five standalone scientific figures and four reports.
 marker=OUT/'07_QC_AND_INTEGRITY/ARTIFACT_OPERATION.json'
 if not marker.exists():
  subprocess.run(['C:/Users/Alban/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe','C:/Users/Alban/.codex/plugins/cache/openai-primary-runtime/pdf/26.1017.11052/skills/pdf/container_tools/mark_artifact_operation_started.mjs','--operation-kind','create','--expected-output-count','9','--output-format','pdf'],check=True)
  jsout(marker,dict(operation='create',expected_PDF_count=9))
 if not (OUT/'06_REPORTS/figures/FIGURE_DATA_PROVENANCE.json').exists():figures(physical,cases,summary)
 def write(path,text):atomic(OUT/path,text)
 pc=physical[['year','policy','S_mwh_row_state','Vmax_MW','overload_hours_engineering','mean_hourly_max_MW','p99_hourly_max_MW']].copy();pc['S_mwh_row_state']/=1e6;pc=pc.rename(columns={'S_mwh_row_state':'S (million)','Vmax_MW':'Vmax MW','overload_hours_engineering':'Hours >1 MW','mean_hourly_max_MW':'Mean V MW','p99_hourly_max_MW':'P99 V MW'})
 annualtable=table(pc)
 agg=[]
 for (mapping,policy),g in cases.groupby(['mapping','policy']):
  valid=g[g.converged];agg.append(dict(Map='Common v2' if mapping=='COMMON_PQ_V2' else 'Historical',Policy=policy,N=len(g),Converged=int(g.converged.sum()),Thermal_pass=int((valid.thermal_violations==0).sum()),Voltage_pass=int((valid.voltage_violations==0).sum()),Q_pass=int((valid.max_q_limit_violation_mvar<=1e-6).sum()),Peak_loading=float(valid.maximum_loading_percent.max()),J_AC_sum=float(valid.ac_apparent_normalized_thermal_exceedance.sum())))
 actable=table(pd.DataFrame(agg));paircols=['mapping','left','right','ac_apparent_normalized_thermal_exceedance_A','ac_apparent_normalized_thermal_exceedance_B','ac_apparent_normalized_thermal_exceedance_TIE','feasibility_BOTH','feasibility_ONLY_A','feasibility_ONLY_B','feasibility_NEITHER']
 pt=pairs[paircols].copy();pt.columns=['Map','A','B','A lower J','B lower J','J ties','Both feasible','Only A','Only B','Neither'];pt['Map']=pt.Map.replace({'COMMON_PQ_V2':'Common v2','HISTORICAL_TWO_MAP':'Historical'});pairtable=table(pt)
 mappingrows=[]
 for policy,g in sensitivity.groupby('policy'):
  valid=g[g.both_converged];d=valid.ac_apparent_normalized_thermal_exceedance_common_minus_historical
  mappingrows.append(dict(Policy=policy,N=len(g),Common_lower_J=int((d< -1e-4).sum()),Historical_lower_J=int((d>1e-4).sum()),J_ties=int((d.abs()<=1e-4).sum()),Max_abs_J_change=float(d.abs().max()),Max_abs_loading_change_pp=float(valid.maximum_loading_percent_common_minus_historical.abs().max()),Changed_feasibility=int((g.common_feasible!=g.historical_feasible).sum())))
 mappingtable=table(pd.DataFrame(mappingrows))
 # Explicit ranking and feasibility changes, with maps kept paired.
 detailed_pairs=pd.read_csv(OUT/'02_MATCHED_AC_VALIDATION/MATCHED_AC_PAIRWISE_COMPARISONS.csv')
 pk=['year','time_index','state_id','pf_case','left','right'];hp=detailed_pairs[detailed_pairs.mapping=='HISTORICAL_TWO_MAP'].set_index(pk);npair=detailed_pairs[detailed_pairs.mapping=='COMMON_PQ_V2'].set_index(pk);rankrows=[]
 for left,right in [('C','U'),('C','ROLLING_M2'),('EC','U')]:
  nh=npair[(npair.index.get_level_values('left')==left)&(npair.index.get_level_values('right')==right)];oh=hp.reindex(nh.index)
  col='ac_apparent_normalized_thermal_exceedance_winner';strict=((nh[col]=='A')&(oh[col]=='B'))|((nh[col]=='B')&(oh[col]=='A'))
  rankrows.append(dict(A=left,B=right,Pairs=len(nh),Strict_J_reversals=int(strict.sum()),Changed_J_class=int((nh[col]!=oh[col]).sum()),Changed_feasibility_class=int((nh.feasibility_class!=oh.feasibility_class).sum())))
 rankframe=pd.DataFrame(rankrows);csvout(OUT/'02_MATCHED_AC_VALIDATION/POLICY_RANKING_MAPPING_CHANGE.csv',rankframe);ranktable=table(rankframe)
 impact=[];ck=['year','time_index','state_id','pf_case','policy'];hc=cases[cases.mapping=='HISTORICAL_TWO_MAP'].set_index(ck);nc=cases[cases.mapping=='COMMON_PQ_V2'].set_index(ck)
 for policy in ('ROLLING_M2','U','C','EC'):
  A=nc[nc.index.get_level_values('policy')==policy];B=hc.reindex(A.index);valid=A.converged&B.converged
  impact.append(dict(Policy=policy,Convergence_changes=int((A.converged!=B.converged).sum()),Thermal_status_changes=int(((A.thermal_violations==0)!=(B.thermal_violations==0)).sum()),Voltage_status_changes=int(((A.voltage_violations==0)!=(B.voltage_violations==0)).sum()),Q_status_changes=int(((A.max_q_limit_violation_mvar<=1e-6)!=(B.max_q_limit_violation_mvar<=1e-6)).sum()),Peak_changes_over_1pp=int((abs(A.maximum_loading_percent-B.maximum_loading_percent)>1).sum()),Max_voltage_extremum_shift_pu=float(np.maximum(abs(A.min_vm_pu-B.min_vm_pu),abs(A.max_vm_pu-B.max_vm_pu)).max())))
 impactframe=pd.DataFrame(impact);csvout(OUT/'02_MATCHED_AC_VALIDATION/AC_MAPPING_IMPACT_SUMMARY.csv',impactframe)
 impacttext=table(impactframe[['Policy','Convergence_changes','Thermal_status_changes','Voltage_status_changes','Q_status_changes','Peak_changes_over_1pp']])
 headtable=table(head[['year','policy','affected_hours','affected_buses','maximum_upward_excess_MW','maximum_downward_excess_MW']].rename(columns={'affected_hours':'Conditional hours','affected_buses':'Buses','maximum_upward_excess_MW':'Up excess MW','maximum_downward_excess_MW':'Down excess MW'}))
 buses=pd.read_csv(OUT/'03_GENERATOR_HEADROOM/BUS_AGGREGATE_NECESSARY_CONDITIONS.csv');missing=buses[buses.source_devices=='[]'][['bus_id','balancing_share','model_generation_lower_MW','model_generation_upper_MW']].drop_duplicates().sort_values('bus_id');csvout(OUT/'03_GENERATOR_HEADROOM/BALANCING_BUSES_WITHOUT_NATIVE_GENERATORS.csv',missing)
 head_context=f'''The fleet-only test fails in every hour of every policy-year. This count is structurally driven in part by balancing shares at buses {', '.join(map(str,missing.bus_id))}, where the reconstructed native generator/interchange fleet contains no device; their combined balancing share is {100*missing.balancing_share.sum():.3f}%. The model also contains separate renewable/load components, so absence from that native fleet does not prove absence of all physical resources. The all-hour failure must not be interpreted as an identified generator violation or as a policy ranking. It demonstrates that the fixed balancing key cannot automatically be interpreted as redispatch of only the inventoried native fleet.\n\n'''
 common_pairs=pairs[pairs.mapping=='COMMON_PQ_V2'].set_index(['left','right'])
 narrative=[]
 for left,right in [('C','U'),('C','ROLLING_M2'),('EC','U')]:
  rr=common_pairs.loc[(left,right)];narrative.append(f"For {left} versus {right}, the common-map AC score favors {left} in {int(rr.ac_apparent_normalized_thermal_exceedance_A)} cases and {right} in {int(rr.ac_apparent_normalized_thermal_exceedance_B)}, with {int(rr.ac_apparent_normalized_thermal_exceedance_TIE)} ties.")
 ranking_narrative=' '.join(narrative)+' The EC annual DC improvement therefore does not establish an improvement in these selected AC checks. These counts concern the fixed stress sample only.\n\n'
 common=cases[cases.mapping=='COMMON_PQ_V2'];hist=cases[cases.mapping=='HISTORICAL_TWO_MAP'];feasible=int(common.ac_feasible_under_declared_checks.sum());thermal=int((common.thermal_violations==0).sum());nonconv=int((~cases.converged).sum())
 maxload=float(common.maximum_loading_percent.max());maxq=float(common.max_q_limit_violation_mvar.max());vlow=float(common.min_vm_pu.min());vhigh=float(common.max_vm_pu.max())
 intro=f'''# Paper B: final engineering validation, version 2

18 October 2026. Frozen policies; no optimization reruns. New declared participant-level common P/Q mapping, with a separate unchanged historical two-map sensitivity.

## Executive scientific verdict

CONDITIONAL GO for the controlled DC accounting claims, with explicit engineering limitations. Historical baseline reconstruction passes. The new common-mapping registry contains 648 cases, and the matched historical-mapping registry contains 648 cases. Across the two campaigns, {len(cases)-nonconv}/1296 cases converge. Under the common map, {thermal}/648 pass the thermal check and {feasible}/648 pass all declared thermal, voltage and simple-Q-bound checks. Peak current loading reaches {maxload:.3f}%. These are selected stress cases, not annual security certification. Generator-specific balancing allocation and reserve/ramp feasibility remain unidentified.

The debit geometry and Gate 1 annual certificates are unchanged. This new AC experiment changes only the participant reactive embedding; it does not alter A, any action trajectory, background P/Q, generator reference dispatch, ratings, contingency, PF assumption or solve setting. Its common-map choice is a declared modeling assumption, not a reconstruction of undocumented device locations.

## Authority, frozen inputs and policies

The publication authority is the hash-verified Gate 3 main manuscript and technical report in `TPWRS_PaperB_Final_Signoff_20261017`, with the new independently reconstructed M1-LP lower bounds and conservative marginal-relief premium. Six authority identities are recorded in `AUTHORITY_IDENTITIES.csv`. The original `Final/specification/MANUSCRIPT.tex` is not substituted. The blocked preflight is retained in its original directory without edits.

The frozen physical case is `2ed58c3ba8dd673cc53996f713acb983126a9df45922afcb2d5adb3c2e34aeb6`, with 45 buses, 114 directional positions, 94 retained electrical states and 8,837 monitored rows per hour. The 2029-2031 evaluations contain 8,760 UTC hours each. All requests, eligibility/action bounds, background injections, A, signed sensitivities, offsets and rating rows retain their original order and hashes.

Annual M1-LP is a relaxation schedule, not an implementable touched-time controller. Annual M2 is the archived selected schedule. Rolling M1/M2, U and C use their original raw implemented actions. EC is the archived retrospective positionwise energy-controlled witness. The raw premium residuals differ slightly from the guarded witnesses printed elsewhere; no raw/guarded trajectories are silently mixed.

## Physical metric definitions and objective interpretation

The original objective is S=sum_t Delta_t sum_state,row [abs(f)-F]_+, in MW h row-state. It sums repeated appearances of overloads in multiple contingency states by definition. It is neither unserved energy, load shedding, probability-weighted risk, simultaneous overloaded MW, nor a dispatch-feasibility test.

The complementary hourly diagnostic is V_t=max_state,row [abs(f)-F]_+. Vmax is its annual maximum. Material overload hours count V_t>epsilon at epsilon=1e-4 MW and 1 MW; 1 MW is a chosen reporting convention, not an official security tolerance. Mean, linear 95th/99th percentiles, intact and contingency maxima/hours, distinct physical elements, and worst row metadata are retained. Distinct elements are deduplicated across states. The maximum across states is not a simultaneous real-world loading.

## Full-year physical comparisons

'''+annualtable+'''
All 21 complete signed-row objective reconstructions agree exactly with the stored independent replay totals. Each object includes all 8,837 rows at all 8,760 hours. Integrated residual and tail/duration metrics measure different outcomes: an annual minimizer of S does not minimize overload-hour count or peak row loading. In particular EC can reduce U's cumulative score while leaving many more hours with a small material violation. A retrospective positionwise scaling may spread residual congestion over time without creating an optimized peak or duration objective. That last sentence describes a possible mechanism; the computed counts and S values are verified, whereas a causal attribution to any one hour is not established.

Figures A-C display the residual/peak relationship, material hours, and unsmoothed exceedance-duration distributions. All axes include their relevant zero baseline. No confidence intervals or independent-year statistical inference are used.

## AC model reconciliation and baseline verification

Let sigma_n=+1 for import and -1 for export. The frozen final active map satisfies A[:,n]=sigma_n(C_P[:,n]-pi), and hence p(x)=p_background-C_P diag(sigma)(v-x)+pi sum_n sigma_n(v_n-x_n). Both maps have nonnegative normalized columns. The common version sets C_Q=C_P; the historical version retains its structural C_Q. Their active injections are identical. For lagging PF 0.98, participant Q=C_Q[-sigma*(v-x)*tan(acos(0.98))]; leading reverses sign and unity gives zero participant Q. Column normalization proves total Q is unchanged. An independent participant-by-participant summation reproduces the spatial remapping difference.

The predeclared baseline consists of all three selected states and three PF assumptions at the earliest selected hour in each evaluation year: 27 historical rolling-M2 AC records. Convergence, violation counts, full bus voltages and all monitored loading values match the archive within predeclared tolerances. Archived Q excess, slack P and normalized overload summaries also match. Nine separately executed DC cases reproduce signed affine row flows. The historical archive lacks complete device-level Q vectors, so only its saved Q summary can be independently matched. New runs retain the complete realized gen/ext-grid Q values and declared limits.

PV generators retain source P, voltage targets and Q limits. Original load/sgen/storage elements are disabled and supplied non-PV injections are represented by synthetic PQ elements. One source external grid remains the slack; it supplies nonlinear losses. This is a fixed-injection AC power flow, not AC-OPF or headroom-constrained redispatch.

The frozen DC F values come from central active-power rating estimates. AC loading uses native current limits (including transformer current loading), not those central MW values. Their source conventions remain unchanged and their row mapping is recorded. The inherited CSV column name `ac_apparent_normalized_thermal_exceedance` denotes J_AC=sum[max(loading_percent/100-1,0)]; with this solver configuration loading is current-based. It is not numerically interchangeable with annual S.

## Matched AC outcomes

'''+actable+f'''
The common-map voltage range is {vlow:.6f}-{vhigh:.6f} pu, and its largest recorded simple-Q-bound excess is {maxq:.6g} Mvar. The source bounds are not full capability curves. Nonconvergent cases, if present, remain separate from zero-violation outcomes. A converged power flow may still have severe thermal overload, and a pass of these selected checks does not establish generator headroom or annual AC security.

## Policy rankings under each mapping

'''+pairtable+ranking_narrative+'''
Each row covers 162 matched hour/state/PF comparisons. A and B are compared only within the same AC mapping, timestamp, state and PF setting. Lower normalized J uses a deadband of 1e-4; loading uses 1e-4 percentage point, voltage deviation 1e-6 pu and Q excess 1e-6 Mvar. All-feasibility counts apply the declared voltage 0.95-1.05 pu, thermal 100%+1e-6 tolerance and Q excess <=1e-6 Mvar. Repeated PFs and contingencies are deterministic sensitivities, not independent trials.

## Mapping sensitivity, kept separate from policy effects

'''+mappingtable+ranktable+impacttext+f'''
Differences above are common minus historical at identical policy actions. Every unity-PF result is unaffected within {acqc['unity_mapping_max_metric_difference']:.3g} in the checked metric values. Nonunity changes reflect spatial participant-Q allocation; they do not change total participant Q or active restrictions. Mapping sensitivity is reported for all policies, including the new premiums, without pooling the two mappings in a policy ranking. Changes in the policy win/loss/tie counts between the two policy-comparison blocks show whether ranking conclusions depend on the mapping.

## Generator headroom and fixed balancing

'''+headtable+head_context+'''
These are CONDITIONAL bus-aggregate necessary-condition results, not identified generator-specific headroom violations. The diagnostic first separates background generation and boundary devices from participant demand/generation and renewable curtailment. It then asks whether assigning the complete fixed balancing component to that same native generator/interchange fleet, with the other resource components fixed, would exceed summed native P bounds. The separate headroom report states the condition, recovered data and limitations.

The actual A supplies bus shares but no generator-specific allocation. For two generators at one bus, adjustments (d,0) and (0,d) can produce the same bus injection and different device headroom use. Therefore bus injections alone cannot identify Delta P_g. If a fleet-only aggregate bound fails, no allocation within that conditional fleet can realize the same aggregate; this follows by summing its device bounds. It does not rule out other unmodeled load/renewable allocations. Passing an aggregate bound cannot establish reserves, market availability, ramps or individual deliverability.

## Information timing and nonanticipativity

The rolling benchmark uses realized controlled-scenario inputs over a deterministic 48-hour window and commits 24 hours. It is distinct from full-year annual foresight. The marginal score uses the requested-state derivative and no optimized future action, but its future-hour requested/background inputs are not authenticated forecast vintages. The card/thresholds are frozen across held-out years; each year's bucket assignments vary with its scenario inputs.

A prospective schedule and rate would have to be measurable with respect to information at the applicable announcement time. Forecast products, issue-time stamps, revisions, network availability and contract announcement/settlement rules would need authentication. No 07:00 implementation, forecasting experiment or prospective controller is validated. Finite-window perfect information is not a general optimal-value upper bound on all prospective controllers; comparable global horizons, domains and nonanticipativity assumptions would be required.

## Numerical QC, reproducibility and scientific scope

The independent QC aggregates every persisted AC voltage, loading and Q-device row and checks all case metrics and paired keys. It recomputes all hourly diagnostic statistics, verifies alternate physical algebra on a declared sample, and checks all 525 critical row flows directly. Complete objective agreement is independently tied to the stored fixed-action replay identities. The preflight, physical, AC, headroom, timing and independent-QC status files distinguish execution success from physical feasibility.

All new work is in the version-2 directory. Per-case AC and per-object physical checkpoints permit restarting without replacing completed results. The default no-argument PyCharm stage is PREFLIGHT. No annual/rolling optimization, card fitting, trajectory repair, generator/rating change, IEEE/Zenodo upload or modification of the old blocked preflight occurs. Consumed files and sealed Gate 1/Gate 3 identities are checked again before final sealing.

## Impact on the TPWRS manuscript and remaining limitations

The contribution remains the entitlement-domain analysis, exact relaxed-hybrid projection and instance-specific strict annual numerical separation. Different feasible domains do not imply different optimal residuals in every instance. The new AC exercise is an engineering qualification, not a new debit rule or proof of contract superiority. C improves on U in the recorded rolling DC experiment, while ordinary M2 remains better in every held-out year. EC remains a common-feasible retrospective control, not a deployable controller or proof that a differentiated contract is necessary.

Keep one concise main-paper engineering paragraph with the declared common-map assumption and the most consequential thermal/headroom/timing limitation. Put detailed tables and mapping sensitivity in the technical report. Proposed source-grounded LaTeX is provided separately; the locked ten-page manuscript is not overwritten. No DOI for the new report is invented. Submission remains CONDITIONAL GO for the controlled-model claims, conditional on accurate model/rating/chronology descriptions and retention of adverse engineering findings. Secure operational dispatch, annual AC feasibility and prospective contract superiority remain unproved.
'''
 intro+='''\n## Evidence-to-code map

| Finding | Isolated script | Frozen input family | New result |
| --- | --- | --- | --- |
| Baseline identity | stage_preflight.py | selected_ac and engineering_selection | BASELINE_AC_RECONSTRUCTION.csv |
| Full-year metrics | stage_physical.py | physical_case arrays and saved actions | ANNUAL_PHYSICAL_METRICS.csv |
| Mapping and AC comparisons | stage_ac.py | ac_inputs, native network, original sample | MATCHED_AC_CASES.csv and AC_MAPPING_SENSITIVITY.csv |
| Balancing scope | stage_headroom.py | device registry, background caches, A | BALANCING_FEASIBILITY_BY_POLICY.csv |
| Information timing | stage_information.py | rolling source, frozen card and experiment records | INFORMATION_STRUCTURE_AND_0700_AUDIT.md |
| Independent QC | stage_qc.py | frozen arrays and completed row tables | INDEPENDENT_NUMERICAL_QC.json |

The full absolute paths, hashes and result links appear in SOURCE_AND_RESULT_TRACEABILITY.csv and the consumed-source inventory; the reproducibility guide defines each root and safe stage usage.
'''
 write('06_REPORTS/FINAL_ENGINEERING_REPORT.md',intro)
 ac=r'''# Matched premium AC validation, version 2

## Setup and identities

All 18 original selected UTC hours, three original state IDs per hour and unity/0.98 lagging/0.98 leading assumptions are retained. Four fixed policies are evaluated: ordinary rolling M2, raw U, raw C and retrospective EC. Two separately labeled mappings produce 648 cases each, 1,296 new campaign solves; the independent 27-case preflight adds 27 historical baseline solves. No additional cases are counted unless actually evaluated. Exact cases and configurations are in MATCHED_AC_CASES.csv and AC_CONFIGURATION_V2.json.

The historical two-map configuration preserves the original reactive structural matrix. Common v2 sets C_Q=C_P_final by an explicit participant-level PF modeling decision. It is not a recovery of undocumented physical device locations. Every other scientific input, including fixed PV reference dispatch and original rating conventions, remains unchanged.

The baseline sample was predeclared before solving. All 27 historical AC records and nine independent DC cases pass. The full 162 historical rolling-M2 records were subsequently reproduced under the unchanged evaluator. Baseline tolerances are voltage 1e-8 pu, loading 1e-5 percentage point, normalized J 1e-6, slack P and maximum Q excess 1e-4 MW/Mvar; convergence and stored violation counts match exactly. The native PV/slack Q bounds and controls are preserved. Complete historical generator-Q vectors do not exist in the comparison CSVs, so the historical Q-summary match is not presented as a full device-vector comparison.

## Injection algebra and the mapping change

For sigma_n=+1 import/-1 export, A[:,n]=sigma_n*(C_P[:,n]-pi). Participant net P is -C_P diag(sigma)(v-x), with counter-injection pi sum sigma(v-x). Participant Q is C_Q[-sigma*(v-x)*tan(acos(0.98))] for lagging, reversed for leading and zero for unity. The new C_Q equals C_P exactly, all columns are nonnegative and normalized, active injections remain unchanged, and summing columns proves conserved total Q. The spatial difference is checked through a separate participant-by-participant summation. Unity input Q is identical, and independently solved unity outcomes agree numerically.

The solver is pandapower 3.2.1 Newton-Raphson, dc initialization then flat only if necessary, voltage angles calculated, Q limits enforced, connectivity checked, 100 iterations and tolerance_mva=1e-8. Settings and source network parameters are unchanged. Original active devices are replaced only as specified by the historical reconstruction: source PVs and one slack retained, synthetic fixed PQ represents remaining net injections. Slack P covers nonlinear losses. No redispatch optimization is introduced.

## Metrics and thermal convention

Line loading is max-end current relative to native current capacity (including derating/parallel factors); transformer loading is the historical current-based runpp convention. The inherited apparent-normalized column name is retained for archive compatibility, but J_AC is the sum of positive excesses of native loading/100 over one. It is dimensionless and differs from the DC sum of MW exceedances against central active-power ratings. Native maximum current exceedances are retained in kA, with explicit terminal limits for transformers.

Bus voltages, counts/deviations outside 0.95-1.05 pu and complete gen/ext-grid Q values versus model Qmin/Qmax are retained. Simple bounds are not complete Q capability curves. Nonconvergence is recorded with attempts and failure cause; a failed solve receives no zero overload score. Thermal, voltage and Q checks are distinct.

## Aggregated outcomes

'''+actable+'''
## Paired policy comparisons

'''+pairtable+ranking_narrative+'''
Differences are A minus B on identical year, time, state and PF, always within one mapping. All per-pair score, loading, voltage and Q differences, win/loss/tie classifications and both/only/neither feasibility labels are in MATCHED_AC_PAIRWISE_COMPARISONS.csv. Nonconvergence is separate. Numerical deadbands are 1e-4 J, 1e-4 percentage point peak loading, 1e-6 pu voltage deviation and 1e-6 Mvar Q excess.

## Sensitivity to participant Q embedding

'''+mappingtable+ranktable+impacttext+'''
Common minus historical differences for all 648 identical policy/hour/state/PF objects are in AC_MAPPING_SENSITIVITY.csv. Unity cases are unaffected apart from numerical noise. Compare policy ranking blocks explicitly to assess reversals or tie changes; do not infer mapping independence from aggregate totals alone. The detailed CSV preserves adverse and favorable changes alike.

## QC, interpretation and verdict

All raw bus/branch/reactive-device rows are independently aggregated against the saved summaries, and all paired keys are unique and complete. Native line loading is independently reconstructed from saved current and current rating. Full historical rolling-M2 comparisons pass the original predeclared reconstruction tolerances. Input hashes and source/code identities remain linked to the configuration.

CONDITIONAL GO for selected fixed-injection AC interpretation only. Lower J is not AC thermal feasibility; convergence is not generator deliverability; repeated contingencies and PF settings are not independent trials. No annual AC-security claim, reserve/ramp verification, prospective policy advantage or AC-constrained optimal-value certificate follows. These results qualify the engineering interpretation while leaving the original DC entitlement geometry and annual numerical certificates unchanged.
'''
 write('02_MATCHED_AC_VALIDATION/MATCHED_PREMIUM_AC_VALIDATION_REPORT.md',ac)
 headtext='# Generator headroom and fixed balancing\n\n## Scientific finding\n\nGenerator-specific policy adjustments are NOT IDENTIFIABLE FROM AVAILABLE DATA. No generator allocation, reserve availability or ramp feasibility is invented. The following conditional aggregate diagnostics are reported separately from implementability.\n\n'+(OUT/'03_GENERATOR_HEADROOM/HEADROOM_DATA_LIMITATIONS.md').read_text()+'\n## Conditional full-year results\n\n'+headtable+head_context+'''\n## Mathematical scope

Summing Pmin_g<=P_g<=Pmax_g at a bus gives a necessary aggregate inequality. A violation of that aggregate precludes every device allocation within that specified fleet and with those fixed other resource components. The converse does not establish an actual allocation, ramp feasibility or reserve availability. A bus with two generators generally permits many device adjustments with the same bus sum; A cannot distinguish them. Native source bounds are model-defined, not independently authenticated availability or reserve products.

## Recovered data and AC treatment

GENERATOR_CAPACITY_INVENTORY.csv records all native device buses, names, service flags, P and Q bounds and the absence of authenticated reserve/ramp information. AC_REFERENCE_GENERATOR_BOUND_CHECK.csv checks the fixed PV reference P against its native bounds; those PV schedules are identical across policies. BACKGROUND_DEVICE_QC.csv binds recovered security-adjusted background dispatch to its cache hash and full physical-case background arrays. Device registry columns distinguish internal generators, fixed signed interchange groups and renewable-downward-flexibility devices. The conditional test never applies P bounds to a net bus injection.

The counter-injection mapping is fixed for normal operation and retained line/transformer states. It is a scheduled-injection assumption, with no generator-outage or corrective-redispatch model. The AC builder holds PV P fixed and creates synthetic PQ injections; its convergence cannot verify active-power feasibility of those unassigned injections. Nonlinear slack P handles losses and is separately recorded, not silently allocated among generators.

## Verdict

CONDITIONAL GO for accounting-model interpretation. The conditional bound results must retain their stated fleet-only scope. Generator-level implementability remains unverified and must not be claimed. The historical optimization model, A and annual certificates are unchanged; no headroom-constrained optimum or corrected policy is computed.
'''
 write('03_GENERATOR_HEADROOM/GENERATOR_HEADROOM_AND_BALANCING_REPORT.md',headtext)
 # Compact proposed main-paper text; full tables stay in this engineering report.
 amendments=r'''% Proposed replacements/additions only; locked sources remain immutable.
% Authority: Final_Signoff_20261017/GATE3_MANUSCRIPT/main.tex.
% Place physical assumptions before the first numerical table.
\paragraph{Physical-model scope.}
The fixed transaction matrix prescribes balanced bus-injection changes;
it does not assign generator-specific balancing actions or impose
generator headroom, reserve, ramping, or AC constraints. The annual
certificates therefore concern the stated soft-thermal DC accounting model.

% At the definition/interpretation of S.
The residual $S$ sums thermal exceedances over monitored element--state
rows and time. A physical overload represented in several contingency
states contributes repeatedly. Consequently, $S$ is neither unserved
energy nor a probability-weighted risk or simultaneous overloaded-power measure.

% Concise engineering results paragraph; detailed tables belong in the report.
\paragraph{Engineering diagnostics.}
We additionally evaluate the hourly maximum element--state exceedance
$V_t=\max_{c,r}[|f^c_{rt}|-F^c_r]_+$ and the number of hours with $V_t>1$ MW.
The 1-MW threshold is a reporting convention. Lower integrated residuals
do not uniformly reduce peak loading or material overload hours.
The retrospective exposure control can lower the uniform policy's
integrated residual while increasing its material-overload duration.

% New AC model, separate from the historical two-map registry.
For a new participant-level constant-power-factor evaluation we set
$C_Q=C_P^{\rm final}$, retaining all other physical inputs and solve settings.
This is a declared common-embedding assumption, not a reconstruction of
undocumented device locations. We replay rolling M2, $U_{1.25}$, $C_q$ and
the retrospective $x^{\rm EC}$ at the original 18 hours, three states and
three power-factor settings (648 cases), and retain a separately labeled
648-case historical-mapping sensitivity. The historical model is unchanged.
The native AC current-loading conventions differ from the DC central
active-power rating convention. Convergence and lower normalized AC
overload do not establish thermal feasibility or annual AC security.

% Headroom scope, after the engineering checks.
The fixed balancing vector identifies bus-level counter-injections but
not their generator-specific allocation. A conditional aggregate-bound
diagnostic is reported separately; it does not establish reserves,
ramps or operational deliverability. The AC reconstruction fixes PV
active powers and assigns remaining bus injections to synthetic PQ
elements, so its convergence does not verify generator active-power headroom.

% Rolling information assumptions.
The rolling experiment uses realized controlled-scenario inputs over a
48-hour deterministic look-ahead and implements 24 hours at each epoch.
The marginal score uses the requested-state derivative without future
optimized actions, but authentic issue-time forecast vintages and
announcement-time measurability are not tested. Neither a validated
07:00 controller nor a universal information upper bound is claimed.

% Discussion/conclusion qualification; preserve existing theory and controls.
The exact hybrid projection characterizes feasible action domains,
whereas the strict annual separation is instance-specific. The observed
premium improvement over the uniform rolling control does not establish
superiority over ordinary M2, which performs better in each evaluation
year. The common-feasible exposure witness is retrospective. The new
engineering results qualify the physical interpretation without extending
the original annual certificates to an AC- or headroom-constrained model.
'''
 write('05_MANUSCRIPT_AMENDMENTS/PROPOSED_MANUSCRIPT_CHANGES.tex',amendments)
 write('05_MANUSCRIPT_AMENDMENTS/MANUSCRIPT_CHANGE_RATIONALE.md',f'''# Proposed manuscript changes

The locked main manuscript is unchanged. Proposed LaTeX addresses physical assumptions before the first numerical table; S and contingency multiplicity; peak/duration diagnostics; distinct historical/common/sensitivity AC evidence; headroom identification; finite-window realized information; and discussion/conclusion scope. No favorable result or actual DOI is invented. The abstract needs no numerical expansion: its existing adverse selected-AC qualification remains supported.

Common-map totals: {thermal}/648 thermal pass, {feasible}/648 all-check pass, peak loading {maxload:.3f}%. These can support one compact engineering paragraph; the detailed tables should remain in the standalone technical report to preserve main-paper mathematical self-containment. The proposed text deliberately does not insert the full 21-row/8-policy-table dossier into the ten-page manuscript. A future report citation must use the actual verified DOI after a separately authorized deposit; no upload occurs here.

The joint P/Q mapping is an author-declared version-2 assumption. Historical results retain the original reactive map. Raw U/C actions remain separate from guarded annual witnesses, and EC remains retrospective. The new AC model supplies no altered annual lower bound or headroom-constrained optimum.
''')
 from prepare_exact_amendments import run as exact_amendments
 exact_amendments()
 guide=r'''# Engineering reproducibility guide, version 2

## Safe PyCharm entry point

Open `C:/Users/Alban/Documents/TPWRS_PaperB_Engineering_Validation_20261018_v2` as a separate project. Use `C:/Users/Alban/anaconda3/python.exe` (Python 3.9.7, NumPy 1.26.4, pandapower 3.2.1). Run `RUN_ENGINEERING_VALIDATION.py` with no arguments. Default STAGE="PREFLIGHT" checks consumed hashes and verifies/reuses completed preflight. Other settings: PHYSICAL_METRICS, AC_REPLAY, HEADROOM, INFORMATION_AUDIT, REPORTS, ALL. ALL executes stages in that order. Completed stage markers prevent repeat solves; per-case AC JSON and per-policy physical checkpoints preserve completed work. An unresolved prerequisite raises an explicit error. No original RUN.py or protected module is imported.

All writes are confined to this version-2 directory. Outputs use write-once atomic creation; a mismatched existing file raises an error rather than replacing evidence. Do not delete stage markers or edit a scientific configuration to force reuse under another identity. To change scientific assumptions, create a new isolated version. Execution journals are distinct from immutable scientific results.

The completed blocked version remains at `C:/Users/Alban/Documents/TPWRS_PaperB_Engineering_Validation_20261018`. The present author decision resolves its joint P/Q interpretation by explicitly declaring C_Q_new=C_P_final. It does not retroactively modify the original AC model.

## Source roots and exact identities

Protected root: `C:/Users/Alban/OneDrive - University of Groningen/Desktop/Final Paper Script`. Archive: `PAPER B/TPWRS_FINAL_ARCHIVE` below that root. Every consumed absolute path, byte size and SHA-256 appears in `07_QC_AND_INTEGRITY/CONSUMED_INPUTS.json` and the final consumed-file CSV. `SOURCE_AND_RESULT_TRACEABILITY.csv` links report findings to frozen sources and generated data. `EVALUATION_CODE_IDENTITIES.json` records isolated script identities. `AC_CONFIGURATION_V2.json` binds mappings and solver settings.

Authoritative main/report sources and PDFs are under `C:/Users/Alban/Documents/TPWRS_PaperB_Final_Signoff_20261017`; exact six identities are in `00_AUTHORITY_AND_INPUTS/AUTHORITY_IDENTITIES.csv`. Both sealed submission ZIPs and all retained Gate 1 certificate artifacts are unchanged. The earlier complete-tree inventory is a historical reference; the final check here covers every consumed file rather than rescanning unrelated files.

## Physical inputs and original code references

Physical metadata/arrays: `IMPLEMENTATION/frozen_inputs/artifacts/real_case/physical_case/FINAL_DUTCH_CASE/physical_case.json` and `physical_case_arrays.npz`. Final participant connections are `PARTICIPANT_MAPPING.json` alongside them. The balancing vector is in `IMPLEMENTATION/frozen_inputs/artifacts/real_case/reduction/REDUCTION_EFFECTS.npz`. Signed state sensitivities, offsets, rating rows and UTC ordering are loaded exactly from the frozen arrays. New exact row mapping is `00_AUTHORITY_AND_INPUTS/EXACT_ROW_MAPPING.csv`.

The native network and central-rating source absolute paths are bound by `IMPLEMENTATION/frozen_inputs/artifacts/real_case/LOCAL_SOURCE_CONFIG.json`. Reactive inputs are `IMPLEMENTATION/frozen_inputs/artifacts/ac_inputs/AC_INPUT_FREEZE.json` and `ac_inputs.npz`. Their source hashes are verified. Source PV generation P, background Q and native generator Q limits are retained unchanged.

Historical code references are `IMPLEMENTATION/source/jobs/selection.py`, `jobs/dc.py`, `jobs/ac.py`, `tpwrs/engineering.py`, `tpwrs/network.py`, `tpwrs/rolling.py`, `jobs/input_build.py`, `jobs/debit_rule_family.py` and `tpwrs/debit_rules.py`. They are inspected, not imported or executed. Native construction/solve functions are independently transcribed in the isolated `engineering_common.py` and `ac_evaluation.py`.

## Stored trajectories and chronological semantics

Annual M1-LP/M2: `RESULTS/annual_primary/year_Y/rho_0.15/annual/RULE/solution_arrays.npz`. Rolling M1/M2: `RESULTS/rolling/m1_m2/year_Y/rho_0.15/rolling/RULE/rolling_arrays.npz`. Raw U/C: `RESULTS/debit_rule_family/year_Y/UNIFORM_PREMIUM/RAW_ROLLING_POLICY.npz` and `CONSERVATIVE_STATE/RAW_ROLLING_POLICY.npz`. EC: `RESULTS/debit_rule_family/year_Y/EXPOSURE_CONTROLLED_ANNUAL_WITNESS.npz`. Y is 2029, 2030 or 2031. No action is repaired, scaled, refit or reoptimized.

The frozen card and validation decision are in `RESULTS/debit_rule_family/FITTED_CARD.json`, `EXPERIMENT_FREEZE.json` and `VALIDATION_DECISION.json`. Card parameters are fixed; rate assignments vary with realized scenario inputs. The finite-window information is not authenticated forecast-vintage information. Detailed timing and the prospective mathematical requirement are in `04_INFORMATION_TIMING/INFORMATION_STRUCTURE_AND_0700_AUDIT.md`.

## Baseline and matched AC registry

Frozen sample: `RESULTS/engineering_selection/SELECTED_POLICY_STATE_CASES.csv`, `SELECTED_HOURS.csv`, `SELECTED_STATE_REGISTRY_FREEZE.json`. Historical metrics: `RESULTS/selected_ac/CASES.csv`, `LOADING.csv`, `VOLTAGES.csv`. The predeclared sample and tolerances are in `00_AUTHORITY_AND_INPUTS/BASELINE_CASES_PREDECLARED.csv` and `BASELINE_TOLERANCES.md`. All 27 cases use the original two-map model. The subsequent full 162 historical-M2 comparisons are separately recorded.

`stage_ac.py` creates 648 common-map and 648 historical-map cases, each with four policies at the same original hour, state and PF. Original source PV/slack controls and Newton-Raphson settings are preserved: dc then flat initialization, angles, Q-limit enforcement, connectivity checking, 100 iterations and 1e-8 tolerance_mva. Failure records preserve attempts and no zero-score fallback is used.

All case JSON records are retained under `02_MATCHED_AC_VALIDATION/cases`. Aggregates include MATCHED_AC_CASES.csv, MATCHED_AC_LOADING.csv, MATCHED_AC_VOLTAGES.csv, MATCHED_AC_REACTIVE_LIMITS.csv, MATCHED_AC_PAIRWISE_COMPARISONS.csv, MATCHED_AC_PAIRWISE_SUMMARY.csv, MATCHED_AC_SUMMARY.csv and AC_MAPPING_SENSITIVITY.csv. Mapping tags must remain attached to every comparison.

## Full-year metrics, headroom and independent QC

`stage_physical.py` processes all 8,837 rows in chunks of 128 hours. It creates ANNUAL_PHYSICAL_METRICS.csv, HOURLY_MAXIMUM_EXCEEDANCE.parquet, CRITICAL_ROWS_BY_POLICY.csv and OVERLOAD_DURATION_SUMMARY.csv under `01_PHYSICAL_METRICS`. All 21 policy-year residual totals agree with stored fixed-action replay evidence. Thresholds are 1e-4 MW and the declared 1-MW reporting convention; quantiles use NumPy linear percentiles.

`stage_headroom.py` recovers background device dispatch from `PAPER B/Final/artifacts/real_case/background/background_Y.npz`, bound to its BACKGROUND_AUDIT_Y.json cache hash, and checks exact equality to the physical background. The immutable device registry preserves generator, interchange and renewable-curtailment labels. Native P/Q bounds and fixed AC PV references are inventoried. Counter-injection shares are bus-level; generator-specific adjustments are not identified. Conditional aggregate-bound results do not become a redispatch feasibility proof.

`stage_qc.py` independently aggregates all persisted AC row tables and hourly diagnostic statistics, checks native line loading from current/rating, checks alternate DC algebra on a stated sample and directly recomputes every retained critical-row flow. Existing independent full-action replay totals supply a separate objective-identity check. No extra optimizer or AC solve is hidden in QC. All scope limits are explicit in INDEPENDENT_NUMERICAL_QC.json.

## Reports, figures and manuscript changes

`build_dossier.py` reads completed data and generates five unsmoothed publication figures and report Markdown. `render_dossier.py` creates four PDFs with the bundled ReportLab runtime, and Poppler page images are visually reviewed. Figures A/B use the annual metrics CSV; C uses the hourly Parquet; D uses AC summaries; E uses the complete case registry. Figure provenance JSON gives these exact paths. Deterministic years/PF/contingencies are not treated as independent statistical trials.

Reports are the final engineering dossier, matched premium AC report, generator-headroom report and this guide. Proposed changes are `05_MANUSCRIPT_AMENDMENTS/PROPOSED_MANUSCRIPT_CHANGES.tex` and MANUSCRIPT_CHANGE_RATIONALE.md. No locked manuscript source is overwritten, and no IEEE/Zenodo publication or DOI reservation occurs.

The final verdict is conditional on retaining the declared controlled-model interpretation, adverse thermal findings and headroom/timing limits. The new reactive sensitivity never changes the original annual DC accounting certificate.
'''
 write('06_REPORTS/ENGINEERING_REPRODUCIBILITY_GUIDE.md',guide)
 subprocess.run(['C:/Users/Alban/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',str(OUT/'render_dossier.py')],check=True)
 jsout(target,dict(status='COMPLETE_PENDING_FINAL_INTEGRITY_AND_VISUAL_QA',verdict='CONDITIONAL GO',PDF_reports=4,figure_PDFs=5,optimization_calls=0))

if __name__=='__main__':run()
