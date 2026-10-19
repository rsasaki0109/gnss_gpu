# D236実走の粗いphase計測 (D238, 2026-10-16)

**距離計算の再利用をすぐ実装する方針を見送り、move処理の内訳を次の調査対象にする。** 実走の全197valid stepを使った中央値で、観測取り出しのexclusive wall spanはCPU 4.62% / GPU 4.47%。move blockの残部はCPU 60.22% / GPU 61.03%だった。これは計測器・schedulingを含む粗い時間配分で、CPU cycleの帰属や確定的な高速化上限/短縮率ではない。今回新しい最適化は採用せず、D236/製品/既定を維持する。

## 目的と実装

前回D237の最悪CP行native化は完全PF-step gate0/7だった。次の小さいloopを推測で置換する前に、従来の合成benchmarkがfixtureにしていた実際の`_extract`と`_lambda_topk`を含めて調べた。

外部cssrlibの実装を確認すると、`geodist`は既にNumba geometry helperを呼ぶ。元の`_extract`には各PR pairでrover ref/j・base ref/jの4呼出しがあり、参照衛星・周波数の重複はある。一方でnative距離計算をPythonの再利用管理へ置き換える追加効果は未測定であり、取り出し全体の今回の時間配分からは優先度を低くする。`_extract` exclusiveにはgeometry以外のsupport/covariance/measurement-row構築等も含まれるため、上記割合をgeodist単独の時間とは解釈しない。

`PhaseProfileStep`はD236のlocal method viewだけにcoarse wrapperを置く。元のmethod code・引数・戻り値・例外はそのまま、各stepに新しいspan recorderを作る。`sys.setprofile`、実runtime/module/classのpatch、数値式の置換は行わない。各spanの開始/終了に`perf_counter_ns`を使い、直接の子span時間だけを親から引く。全exclusive spanの整数ns和がroot intervalに厳密一致することを毎step assertする。未計測処理・wrapperの集計overhead・scheduling delayは親/`step`のexclusiveに残る。

inclusiveは子を含み重なる。exclusiveだけがrootを分割する。`_moves`のexclusiveはextend/shrink等の計測済み子を除いた**staging、native scoring/packing、proposal/weight/choice等の残部**で、それらの内訳はまだ分かれていない。`_conditional_mean` exclusiveと`_ls_offset`は別の行で、両方のinclusiveを足さない。合成fixtureで上書きされたextract/searchはwrapせず、その時間を実入力の計測と取り違えない。

粒子/候補/prior/FP64/2IRLS/閾値、conditional/FIX/feedback/graph/RNGは不変。D226数値DLLとD233/D234/D235拡張を再利用し、新しいnative/CUDA build・buffer・H/DD cacheはない。strict NumPy policyは従来のliteral step fallbackを保持し、error時もfinallyで計測stackを解放する。

## 実走の時間配分

Tokyo run3/seed20260710/400epoch、元cluster_relin0のCPU/GPUを各1prefix実行した。各200step（197valid/3skip）、計400paired step。元の実extract・整数探索・RTK callbackを使い、GTSAM graph solveの外側で呼ばれるPF-stepを計測する。単発・非占有・通常priorityのwall spanであり、CPU/GPU速度比較には使わない。

全197valid stepを使用し、cold/slow/negative条件の事後除外・有利な区間の選択を行わない。表はstepごとのexclusive時間/そのstepのroot時間の比の中央値と、exclusive時間自体の中央値で、両中央値の比ではない。rootのexclusiveは未分類残部で、root全体の時間ではない。全root時間はrawの`step.inclusive_ns`へ保存した。

| phase | CPU median exclusive ms | CPU median exclusive fraction | GPU median exclusive ms | GPU median exclusive fraction | calls/backend (200steps) |
|:---|---:|---:|---:|---:|---:|
| `_conditional_mean` | 0.343 | 0.94% | 0.321 | 0.85% | 938 |
| `_ddpr_offset_impl` | 0.235 | 0.58% | 0.189 | 0.55% | 197 |
| `_extend` | 1.792 | 4.55% | 1.614 | 4.43% | 2110 |
| `_extract` | 1.776 | 4.62% | 1.549 | 4.47% | 200 |
| `_feedback` | 0.867 | 2.04% | 0.718 | 1.97% | 197 |
| `_lambda_topk` | 0.642 | 1.67% | 0.601 | 1.68% | 197 |
| `_loglik` | 0.461 | 1.10% | 0.366 | 1.03% | 197 |
| `_ls_offset` | 2.497 | 6.64% | 2.302 | 6.12% | 832 |
| `_maintenance` | 0.092 | 0.19% | 0.076 | 0.18% | 200 |
| `_maybe_resample` | 0.053 | 0.13% | 0.044 | 0.12% | 197 |
| `_moves` | 25.907 | 60.22% | 22.426 | 61.03% | 197 |
| `_shrink_worst` | 1.586 | 3.96% | 1.342 | 3.64% | 2110 |
| `_spawn_candidates` | 0.000 | 0.00% | 0.000 | 0.00% | 1 |
| `step` | 4.508 | 11.46% | 3.910 | 10.93% | 200 |

この結果はmove blockの細分化（staging/score/weight/choice）を優先する根拠として使う。まだどの処理が新しい最適化で何%速くなるかは確定していない。過去D230 proposal再利用やD237 shrinkの失敗を、この表の大きな値で採用へ読み替えない。

## 計測器の追加コスト

同じD236完全stepをreference/profiled/A/Aの3modeで直列回転し、3条件×5round×3group×3mode=135timed calls、9warm snapshotを保存した。自processの1許可CPUへpin、BLAS1thread、通常priority、host/GPU非占有。restore/callback binding/hash/検証は時間外。全callの状態・RNG・既存metadataは同じreferenceに一致する。

round内paired slowdown中央値→5round中央値を用い、p50同士の比とは異なる。abs A/A中央値5%未満かつabs観測overhead5%以下を補助flagにする。最適化の速度gateではない。extract/searchがfixtureなので実走のその2wrapper分はこの較正に含まれない。

| particles/initial_unique/rows/groups | reference p50 ms | profiled p50 ms | paired slowdown | A/A slowdown | low-observed-overhead flag |
|:---|---:|---:|---:|---:|:---|
| particles64/initial_unique6/rows64/groups4 | 19.221 | 19.920 | +2.951% | -1.205% | PASS |
| particles64/initial_unique32/rows64/groups4 | 33.109 | 30.247 | +5.306% | +3.499% | FAIL |
| particles128/initial_unique128/rows128/groups8 | 542.952 | 537.453 | -3.171% | +2.054% | PASS |

低overhead補助flagは2/3、A/A中央値flagは3/3。64粒子/32初期仮説の+5.31%は5%外なので残す。−3.17%は計測器による高速化とは扱わず、個別roundの広い揺れもrawへ保持する。これは粗い候補選定用で、phase比率の精密な帰属やmodel/device/全routeの利得を保証しない。全測定は1回で、失敗条件を再測定しない。

## 正確性と保存

19tests PASS、既知設定warning1。3backend×2seed×3構成×4epochで、D236と全保存state/位置/P/RNG/graph events/既存caller metadataが一致する。nested spanの非負値、root整数ns分割、stepごとの新しいdict、strict-policy途中state/例外とstack cleanupを確認した。

実走はCPU/GPU各200、計400stepで、凍結D236の同backendの保存state JSONL全行・軌跡NPZ全配列dtype/shape/bytes・全cfg・従来全caller metadataが一致した。source/input/external tc/3拡張/数値DLLのhashを照合する。新しい数値許容差は追加しない。Python scalar NaN payloadをcanonical化する既存codecの制約は残る。1route/seed/default設定のprefixで、全route/multi-seed精度・whole-graph速度gateは未完了。

較正135timed/9warm状態も一致する。新しいkernel/メモリ処理がないため、今回はCUDA checkerを繰り返さず、元のD236/D237検査結果を維持する。計測器の時間を製品の高速化とは扱わない。

`last_profile`へ残るのは各phaseのcall数/inclusive/exclusive整数nsだけで、入力H/配列/graphを新たにcacheしない。collectorは診断runnerの外部リストへ各stepを保存する。heap peakは実測しておらず、Pythonのdict/int/closure/stack等は追加される。既存score workspaceはdevice/host各1,143,352bytes/contextで、較正/各実走processのscorerは1個。永続native/CUDA workspaceは増えない。

36production sources/5installed extensions/PF build PE、36prior reports/17archives/2060filesを照合して保持する。

## 利用と記録

```python
from rbpf_phase_profile import PhaseProfileStep

call = PhaseProfileStep(runtime, scorer, 'native_cpu',
    extension=canonical_extension,
    parent_extension=parent_extension,
    mass_extension=component_mass_extension)
runtime.step = call
output = call(rtk, ctx)
spans = call.last_profile  # separate from unchanged call.last
```

source/method guardsが同じ凍結modelだけを受け付ける。scorerは呼出側が所有し、`scalar`/`native_cpu`/`gpu`は従来の数値backend。actual helperは`bench_rbpf_phase_profile_field.py`、追加コストhelperは`bench_rbpf_phase_profile.py`。既存測定を上書きせずfresh outputを使う。

[全raw/独立監査](../benchmarks/rbpf_phase_profile_20261016.json)、[完了照合](../benchmarks/rbpf_phase_profile_completion_check_20261016.json)。私有raw/archiveは`dist/rbpf_phase_profile_*20261016*`へ保存する。製品/既定は維持する。
