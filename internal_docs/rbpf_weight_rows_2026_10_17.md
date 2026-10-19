# RBPF粒子行ごとの重み演算一括化 (D266, 2026-10-17)

完全PF screenはCPU 0/8、GPU同backend 0/8、GPU joint 0/8。明示caller実験として結果を保存し、D236/製品/既定を維持する。前回D265の属性/ID cacheとは別の変更で、D230の共通proposal/probability再利用やD264のextend変更も併用しない。

## 実装と一致の範囲

候補登録/scoring/burst混合まで元コードを保持し、粒子ごとに作っていた重み用tuple/list/NumPy arrayとufunc呼出しを、同じ候補数の粒子行でまとめた。各行の全prior・score・候補・並び・正規化・log/exp/max/sum/除算を毎回計算し、値を粒子間で再利用しない。候補数を合わせるpadding、近似、並べ替え、観測/粒子/候補の削減はない。候補数の違う行は別の連続2D arrayへ入れる。行の候補順は元のまま、和は各行の連続fast axis。

元の `mx + math.log(tot)` とlogw更新、各粒子の `rng.choice`、選択H/tag、commitは元の粒子順。乱数を一括生成しない。FP64・尤度DLL/IRLS/pivot/prior/全閾値・FIX/feedback/resample/graphを維持する。CUDA移植はなく、GPU前後でも動くCPU側処理の実験である。

[NumPyの和の説明](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)では、fast axisのpartial pairwise sumと、他の条件によって精度が変わる点が述べられている。本案の1D/2D bits一致は仕様だけから一般化せず、記録したNumPy 2.3.4 / Python 3.12.10上で検査した。別NumPy build/CPU dispatch/将来versionへの保証ではない。数値許容差は追加していない。

通常のNumPy関数/組込みbinding、Particle slot、標準Generator/BitGeneratorとbuiltin floatの領域を確認する。非標準型/乱数/関数binding/空行/範囲外index/容量超過は元loopへ戻る。全行を副作用のない数値probeで事前確認し、無効prior/非有限/正規化underflow/overflowは元loopへ戻して元の警告・途中logw・RNG・例外を再現する。probe中の警告抑制だけで無効入力を受理しない。厳格なNumPy policyは親controllerの元step fallbackのまま。

4つの元helperのASTは選択loopと3つのcalleeだけを明示置換し、戻すと元ASTへ一致する。元/literal-tail clone/一括版を用意し、全callee/scope refresh/guardをtimer内へ含める。既存metadataは全て保持し、新counterは別項目として比較する。

97tests PASS、既知pytest設定warning1。22候補数×3seed×19粒子行の1254行でmx/tot/全確率のuint64 bitsを元1D式へ比較し、ragged/重複候補/極端scoreも比較した。無効prior・非有限・custom float・空行・index/容量境界、custom RNG/NumPy bindingと途中例外/警告/状態を照合した。scalar/native CPU/GPU×2seed×3構成×4epochで元/literal/一括の全状態・RNG・graph・従来metadata一致。計18case×3variant×4epochで、fixtureの観測/整数探索/graph-solve境界を超える測位精度試験ではない。

## 完全PF比較

8構成、元/同じ新callerのliteral control/一括版をCPU/GPUそれぞれ、CPU/GPU A/Aの8mode。4RNG初期状態seed20260710〜13、5round×3group。3840timed/256warm。全modeでregister-LU scorer context1個を共有し、数値PEは前回と同じ。4stepのinteger-ns合計/group→3group中央値→5round中央値。CPU/GPU同backendは元とliteral両方へ10%以上、改善round4/5以上、対応A/A abs中央値5%未満。GPU jointは一括CPU比と両A/Aも要求する。全負値/不利/揺れ/校正失敗を保存し、完了した測定は取り直さない。

| particles/unique/rows/groups,cluster,feedback | CPU対元 | CPU対literal | GPU対元 | GPU対literal | GPU対候補CPU | CPU A/A | GPU A/A | CPU/GPU同backend/joint |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0: 64/6/64/4,c0,fb1 | +0.519% | +1.570% | +10.218% | +11.655% | -14.537% | -2.786% | +5.857% | False/False/False |
| 1: 64/32/64/4,c0,fb1 | +8.801% | +13.261% | -5.981% | -2.582% | -10.053% | -9.503% | -16.812% | False/False/False |
| 2: 64/64/64/4,c0,fb1 | +3.473% | +9.088% | -4.006% | -4.223% | -15.315% | -5.387% | -4.914% | False/False/False |
| 3: 128/32/64/4,c0,fb1 | +11.685% | +10.505% | +9.133% | +7.508% | -18.345% | +8.293% | -1.135% | False/False/False |
| 4: 64/6/128/8,c1,fb1 | +11.957% | +11.852% | +7.553% | +7.922% | -0.754% | +7.368% | +7.231% | False/False/False |
| 5: 128/128/128/8,c1,fb1 | -6.594% | -6.755% | -0.942% | +4.696% | +7.189% | -5.934% | +3.656% | False/False/False |
| 6: 64/6/64/4,c0,fb0 | +0.612% | +12.060% | +23.531% | +9.896% | -4.640% | -12.036% | +8.641% | False/False/False |
| 7: 64/6/64/4,c1,fb1 | +9.746% | +8.318% | +0.137% | +12.144% | -16.789% | -1.539% | +7.843% | False/False/False |

許可CPU1本/BLAS1/normal priority/非占有、modeは直列でrotate/reverse、seed順もrotate。guard/row分類/数値probe/全allocation/scoringとGPU転送・完了待ち/元の各choice/conditional/FIX/feedback/resampleを含む。restore/deepcopy/RNG生成/callback再bind/hash/検証/gzipは外。全4096raw記録の順序/件数/ns/hash/counterと256warm全state/metadataを独立auditし、全paired gateを再計算した。時刻・衛星・extract/search/graph solveはfixture境界で、実RTK/FGO/route速度やGPU化率へ換算しない。

## 実データ・メモリ・保存

速度screen通過条件が0のため、事前方針どおり追加の実データ400/800状態再走は実施しなかった。今回の97試験と完全PF fixtureでの一致を、実データ測位精度やfullroute/multiseedの証明へ読み替えない。

最大131072cell/callでfallback。返却確率の保持payloadは8×cell bytes、call内だけで、別途prior/likelihood/log-weight等の一時NumPy arrayを持つ。Python/NumPy heap peakは未測定。既存numeric contextのhost/device各1143352bytes、追加CUDA source/転送/永続bufferは0。新しいnative build/sanitizer実行はなく、以前のASan/CUDA検査件数を今回の試験へ加算しない。scalar NaN codecの既存制限は継承する。

36製品source・5installed PE・PF build PE・92過去reports・45archives8013filesを保全。新source/試験/plan/log/全raw/全warm snapshots/独立auditと旧manifestを新archiveへ保存し、公開機種名/UTF8/EOL/Ruff/compile/diff/artifact policyを確認する。製品採用/既定変更/全route・multiseed精度・FGO全体速度の主張はない。

[結果](../benchmarks/rbpf_weight_rows_20261017.json)、[完了照合](../benchmarks/rbpf_weight_rows_completion_check_20261017.json)、[実装](../benchmarks/rbpf_weight_rows.py)。
