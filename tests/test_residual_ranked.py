from experiments.run_cpu_residual_ranked import calibrate,PROFILES,N_CAL


def test_rank_and_common_gate_preserve_all_profile_constraints():
    rows=[dict(profile=p,score={'bic_improvement':float(r+offset)}) for p,offset in zip(PROFILES,[0,1000,2000]) for r in range(N_CAL)]
    result=calibrate(rows)
    assert all(s['ranked']['rank_from_largest']==2 for s in result['strata'].values())
    assert sum(s['ranked']['actual_failure_bound'] for s in result['strata'].values())<=.05
    assert result['gates']['ranked']['score']==2638
    assert result['gates']['maximum']['score']==2639
    assert result['gates']['ranked']['calibration_count']==1920
