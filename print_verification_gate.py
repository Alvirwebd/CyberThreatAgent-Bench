import pandas as pd

print('======================================================================')
print('PHASE 4 VERIFICATION GATE OUTPUTS')
print('======================================================================')

# 1. Error Taxonomy Breakdown
print('\n[1] Stage 3 Error Taxonomy (artifacts/errors_stage3.csv):')
df_err = pd.read_csv('artifacts/errors_stage3.csv')
print(df_err['error_type'].value_counts().to_string())

# 2. Stage 7 Causal Entity-Temporal Edge Metrics
print('\n[2] Stage 7 Attack Graph Metrics (artifacts/stage7_graph.csv):')
df_g = pd.read_csv('artifacts/stage7_graph.csv')
print('Mean Tech Node Precision : %.4f' % df_g['tech_node_P'].mean())
print('Mean Tech Node Recall    : %.4f' % df_g['tech_node_R'].mean())
print('Mean Edge Precision      : %.4f' % df_g['Edge_P'].mean())
print('Mean Edge Recall         : %.4f' % df_g['Edge_R'].mean())
print('Mean Edge F1             : %.4f' % df_g['Edge_F1'].mean())
print('Host Node Accuracy       : %.4f' % df_g['host_node_correct'].mean())

# 3. Stage 8 Risk Calibration & Separation
print('\n[3] Stage 8 Risk Calibration (artifacts/stage8_risk_ablation.csv):')
df_abl = pd.read_csv('artifacts/stage8_risk_ablation.csv')
print(df_abl.to_string(index=False))

# 4. Agent vs Pipeline Comparative Evaluation
print('\n[4] Complete Agent vs. Pipeline Comparison (artifacts/agent_vs_pipeline_eval.csv):')
df_ag = pd.read_csv('artifacts/agent_vs_pipeline_eval.csv')
agg = df_ag.groupby('regime').agg({
    'technique_f1': 'mean',
    'mitigation_ndcg3': 'mean',
    'false_escalation': lambda x: float(pd.Series(x)[df_ag.loc[x.index, 'is_malicious'] == 0].mean()),
    'tool_precision': 'mean',
    'redundant_call_rate': 'mean',
    'latency_ms': 'mean',
    'token_cost': 'mean',
}).reset_index()
print(agg.to_string(index=False))
print('======================================================================')
