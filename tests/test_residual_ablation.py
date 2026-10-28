from experiments.run_cpu_residual_ablation import calibrate


def test_same_paired_calibration_separates_model_and_coverage():
    rows=[dict(profile=p,old_score={'bic_improvement':float(old)},score={'bic_improvement':float(new)})
          for p,old,new in [('white',1,2),('center',5,4),('edge',3,1)] for _ in range(408)]
    g=calibrate(rows)['arms']
    assert {a:v['score'] for a,v in g.items()}=={'unweighted_white':1,'unweighted_multi':5,'weighted_white':2,'weighted_multi':4}
    assert g['weighted_white']['calibration_count']==408
    assert g['weighted_multi']['calibration_count']==1224
