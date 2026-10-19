# RBPFの同一候補生成を再利用する (D228, 2026-10-16)

凍結公開RBPF向けの明示`MoveReuseStep`を追加した。D227のmove/weight/choiceコードをそのまま使い、同一エポック内の同じ整数辞書に対するextend/shrinkの結果だけを再利用する。全粒子・全move・重複candidateのprior・要求順・乱数順を維持し、数値式や閾値は変更しない。公開凍結runtime・製品source/PE・既定設定は変更しない。

元のextendは同じHとprobだけから条件付き丸めを行い、shrinkは同じCP misfitから1つのsatを落とす。いずれもRNGや粒子weightを読まず、入力Hを変更しない。キーにはgroup/satelliteの挿入順と絶対整数値を残す。canonical basin IDで別gaugeや別順序の辞書を共有しない。通常のbuiltin int/dict以外は元methodをそのまま呼ぶ。
per-call viewだけが2つのmethodを包み、実runtimeのmethodを一時差し替えしない。`None`も結果として保存する。結果は元のcommitまでread-onlyで共有され、選ばれたHは元commitが粒子ごとにdeep copyする。cacheはmove blockを出ると破棄され、次epochの観測・Qb・CP misfitには持ち越さない。D227のsource/method guard、strict-policy完全step fallback、全候補scalar fallbackを引き継ぐ。

## 完全stepの合成比較

6条件×5round×3group×7mode、630 timed complete steps。元scalar、既存C++/GPU caller、再利用scalar/C++/GPU、同一既存C++ A/Aを回転比較した。再利用3方式の対照比較270、A/A90。graph抽出・integer-search・衛星/時刻入力だけがfixtureで、候補生成/key作成・pack/全転送/返却・条件付き位置・mass/FIX・resamplingは時間内。setup/restore/warm/検証/保存は時間外。
自processの1許可CPU・1BLAS thread、通常priority、非占有。paired短縮率はround内中央値→5round中央値。C++再利用のscreenは既存C++比10%以上・4/5round改善・A/A中央値abs5%未満。GPU固有screenは再利用C++比にも同条件を要求する。全raw・失敗・roundを残し、合格のための再試行はしない。

| 粒子 / 初期固有H / rows / groups | 既存C++ ms | 再利用C++ ms | paired短縮 | 改善round | A/A短縮 | screen |
|:---|---:|---:|---:|---:|---:|:---|
| 64 / 6 / 64 / 4 | 31.594 | 20.612 | 29.58% | 5/5 | +0.43% | PASS |
| 64 / 32 / 64 / 4 | 60.540 | 57.049 | 5.92% | 4/5 | -7.45% | FAIL |
| 64 / 64 / 64 / 4 | 46.065 | 49.021 | -14.47% | 2/5 | -11.81% | FAIL |
| 128 / 32 / 64 / 4 | 124.835 | 87.477 | 23.87% | 5/5 | -6.26% | FAIL |
| 64 / 6 / 128 / 8 | 67.130 | 41.508 | 36.11% | 5/5 | +2.34% | PASS |
| 128 / 128 / 128 / 8 | 1149.192 | 1033.611 | 3.29% | 4/5 | -2.58% | FAIL |

**重複が多い2条件で約30%/36%短縮を確認した。全条件で速いとは扱わない。** 中程度の重複でもA/A不合格なら効果認定しない。独立p50同士の比からpaired screenを書き換えない。GPU対再利用C++は全6条件で不合格であり、GPU既定化は行わない。scalar A/AでGPU固有の全変動を診断できるとは扱わない。
全630callで同じbackendの既存callerと保存state/離散/RNG/条件付き位置/Pがbit一致し、scalar再利用はliteral元実装とbit一致した。再利用の変更について数値許容差を設けて合格させていない。C++/GPU尤度と元NumPy自体の算術差はD226/D227のままで、別backend同士のbit一致を主張しない。

## 実走での一致と限界

custom GTSAM/cssrlibを使い、run3/seed20260710の400epoch、run2/同seedの400epoch、run3/seed20260711の800epochを既存C++/再利用C++で実行した。3つのprefixで合計1,000回の実PF step、有効990回。保存した内部state・全軌跡NPZ配列が同じbackend間でbit一致した。位置/P/FIX/particle/RNG/weight/feedbackを含む。別途run3/seed20260710の再利用GPUも、D227の凍結GPU200stepと全保存stateが一致した。
NumPy配列は全bytesを保存し、Python scalar NaN payloadはD227のJSON codecでcanonical化する。そのNaN payloadまでの一致は主張しない。C++と元NumPyの追加比較はrun3/seed20260710の200PF stepだけで、位置/P/離散/RNGは一致、その他連続値には最大5.97e-13差がある。複数prefixを全route精度gateの完了とは扱わない。

| 実走prefix | PF step | 各extend/shrinkの既存call → 再利用call | call削減 | 既存C++ PF秒 → 再利用C++ PF秒 |
|:---|---:|---:|---:|---:|
| run3 / seed20260710 / 400epoch | 200 | 12,504 → 2,110 | 83.13% | 13.511 → 13.138 |
| run2 / seed20260710 / 400epoch | 200 | 11,816 → 1,541 | 86.96% | 10.869 → 9.645 |
| run3 / seed20260711 / 800epoch | 600 | 38,041 → 6,501 | 82.91% | 37.858 → 29.447 |

これはunpinned/非占有の各1回の直列実走で、初回compile/cache・順序・負荷の影響を除く統制比較ではない。PF callだけの時間で、hash/snapshot保存は除外している。約3%〜22%の時間差を実走で確立した速度利得とは扱わない。runner全体のwall差も効果認定に使わない。最初のrun3 prefixではcallを83%減らしても全体時間差は小さく、候補生成だけが全体を支配しているとは言えない。
全3prefixで全particle/move数とprior cache hit数を維持し、非標準型bypassとscalar fallbackは0だった。1,041,720要求moveを削らず、各extend/shrinkのcallは合計62,361→10,152。入力データhash、公開source、外部tcのPython source、native DLLを前後照合した。

別の既存C++400epoch実走でPython call observerを試した。canonical key生成238,763call、DD展開55,551callを確認した。ただしobserver付きPF時間は約83秒で、非instrumented約13.5秒の約6倍となった。**この時間内訳から通常実行のhotspot比率や利得を推定しない。** 回数は次のkey/DD重複除去候補を絞る補助であり、低侵襲な切り分けが別途必要。観測runも200stepの全保存stateが同じ既存C++ prefixと一致した。

## 検証と利用

新65tests PASS、pytest設定warning1。3seed×4epoch、cluster/unheld uniform/Gaussian/IRLS有無、shadow/spawn、resampling、maintenance、strict例外の部分state、feedback commit/releaseを比較した。gauge/辞書順の異なる4種類をcanonical keyなら同一になるケースで区別し、実method callが64→4になることをprofileで確認。`None`の再利用、epoch更新、非標準integer型のbypass、commit後の粒子辞書非aliasも検査した。
新caller memcheckは10 targeted tests PASS、warning1、0 error/0 leak。native CUDA source/DLLはD226と同一で再buildなし。以前のracecheck/synccheckを新規PASSに足していない。
測定終了後、benchmarkの時間外hash用lambdaだけを同等のnamed functionへ変更してRuffを修正した。測定時sourceを別保存し、timed runのAST一致と前後hashを照合した。再測定や結果の上書きはしていない。

```python
from rbpf_likelihood_batch import BatchScorer
from rbpf_move_reuse import MoveReuseStep

saved_step = pf.step
with BatchScorer(explicit_dll_path) as scorer:
    pf.step = MoveReuseStep(pf, scorer, mode="native_cpu")
    try:
        result = pf.step(rtk, epoch_context)
    finally:
        pf.step = saved_step
```

再利用の利益は実際のHの重複と候補生成の時間比率に依存する。多様性が高いpopulationではkey作成とmemoの費用が増える。cacheの上限はそのmove block内のheld particle数で、全epoch履歴を保持しない。validな有限入力と凍結pure methodが実験domainで、同一filterへの並行呼出しや任意callbackの副作用を含む汎用置換ではない。

再現: [合成比較](../benchmarks/bench_rbpf_move_reuse.py)、[実走wrapper](../benchmarks/bench_rbpf_move_reuse_field.py)（既存runnerの引数に`--reuse-moves`を加える）。記録: [全結果](../benchmarks/rbpf_move_reuse_20261016.json)、[完了照合](../benchmarks/rbpf_move_reuse_completion_check_20261016.json)。
