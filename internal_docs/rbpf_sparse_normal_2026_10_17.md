# D250: GPU正規行列の構造的ゼロ積を省く

2026-10-17。観測グループ別の行indexを使い、正規行列/RHSの0との積を省くGPU kernelを実装した。全scoreの短縮は旧GPU比-0.08/+0.63%、同kernelのdense版比+0.52/+0.85%に留まり、共同速度gate0/2。製品・既定への採用は見送る。転送は元D226のpageable/synchronousのままで、D248/D249は併用しない。

## 同じ値を保つ条件

position3列同士のdotは全kept rowを元順で加算する。group列とのdot/RHSは、そのgroupに属するkept rowのindexだけを元順で加算する。異なるgroup列同士は構造的に0になる。full asymmetric priorとの加算、pivot/tie/LU/solve、最大2IRLS更新、全観測・候補・prior・閾値・RNGは保持する。triangle copy、近似逆数、epsilon pruning、tree reduction、対称化/jitterは追加しない。CPUは元のdense bodyを使う。

各stageでAw/bwの全要素を確認し、有限かつabs<=2^500、非所属group列がpositive-zeroのときだけ省略する。最大128rowなので、dotの各積と全partial sumは有限に収まる。FP64/RNE・fmad=falseでpositive-zeroから始める元の加算列から、signed-zeroの加算だけを除く。非ゼロ項の順序を変えない。この範囲でdotの結果を保つ構成とし、範囲外・非有限値・想定外の構造/ゼロ符号は元normal_entryへ戻す。unsafeはそのhypothesisの後続stageでも解除しない。priorは元処理のままで、そのNaN/overflow処理へ手を加えない。
[CUDA12.8 FP64丸め](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)、[浮動小数点とcompiler flags](https://docs.nvidia.com/cuda/archive/12.8.1/floating-point/index.html)。

元数値prefix/CPU bodyはliteralコピー。新DLL内の元kernelも5144normalized SASS/opcode/operandが旧DLLと一致する。新sparse/dense kernelは6440命令、62register/shared31456B/stack0/spill0。元は56register/shared30904B/stack32B/spill0。raw normal試験kernelは別で、通常scoreからは呼ばない。命令数や省略積数を速度利得と読み替えない。

## 数値・状態の試験

main147＋追加6の計153tests PASS（各suiteに既存pytest asyncio設定warning1件）。元DLL・同DLL元kernel・新dense・新sparseの同backendでscore/normal/solution/logdet/status bitsを比較し、3seed×3size×6options、odd count/rows0/1/63/127、元のfallback/例外/警告、24workspace lifetime/返却array所有権、3seed×4PF設定×4epochの全保存state/RNG/graph eventを確認した。

raw normal/RHSの直接比較は2292行列。1440件はn3〜11×rows0/1/17/64/128でsigned-zero/subnormal/非対称prior/overflowを含む。84件はNaN/Inf/RHS/magnitude/構造/negative-zero/不正groupの元denseへのfallback。追加768件はAw/bwを必ずguard内に置き、全件でindexed分岐を実行させたsigned-zero、最小subnormal、桁差、ordered cancellation、極端なpriorの比較。両返却のdtype/shape/bytesが一致した。追加suiteも元147test sourceも実行後に書き換えていない。

CUDA memcheck/racecheck/synccheckは各17対象subsetでPASS、error/hazard/leak0。追加suiteのpytest fixture importと同名引数によるRuff F811だけを対象file単位で除外し、その他は所定Ruff/compile/EOL/diff checkを通す。初回lintのF811は試験失敗ではなく、再実行やsource改変で隠さない。CUDA allocation/非同期故障の注入試験は行っていない。

## 実入力の速度と実際に省けた仕事

D243のCPU実走から保存したcluster0/1各197batch、全394入力を使う。8mode（最適化CPU、旧GPU、同DLL元kernel、新dense、新sparse、旧GPU A/A、CPU A/A、新sparse A/A）×5round×3group、47280timed/3152warm。全warmの同backend score/diagnostic/status/inputと全timedの結果hashが一致した。GPU同士の比較に別backendの丸め差を混ぜない。

workspace/warm/復元copy/hashはtimer外。dedup/DD/cluster/pack、group index作成、全finite/structure guard、元の全転送/kernel/同期返却、fallback/scatterはtimer内。GPU処理は直列、allowed CPU1本/BLAS1/normal priority/非exclusive。各group全197batchの整数ns合計→round内3group paired率median→5round median。事前gateは旧GPU・同DLL元GPU・新dense・CPUの全てに10%以上、各4/5round以上正。旧GPU/CPU/sparse A/Aと同DLL元GPU対照のabsmedian<5%。新denseにはコード/resource差があるため、その旧GPU比は診断値で、等速対照の条件にはしない。全rawを保持し、完了した候補の再測定はしない。

時間列は197batch合計median÷197で、個別batchのp50ではない。短縮率は別のpaired統計量。

| cluster | CPU ms | 旧GPU ms | 新dense ms | 新sparse ms | 対旧GPU短縮 | 対dense短縮 | 対CPU短縮 |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.566 | 3.444 | 3.459 | 3.421 | -0.083% | +0.523% | -137.064% |
| 1 | 2.356 | 4.220 | 4.260 | 4.225 | +0.627% | +0.854% | -79.925% |

| cluster | GPU A/A | CPU A/A | sparse A/A | 同DLL元GPU対照 | controls |
|:---:|---:|---:|---:|---:|:---:|
| 0 | +0.396% | -0.120% | -0.945% | +0.406% | True |
| 1 | -0.571% | -0.647% | -1.090% | -0.230% | True |

全speed計測後に394入力をdense/sparseへ別途適用し、計788callでstage/省略積counterを記録した。全返却hashはwarmと一致する。通常のtimed scoreではcounter bufferを使わず、追加のcounter転送を行わない。

| cluster | sparse stage | 省略した0との積 | unsafe dense stage |
|:---:|---:|---:|---:|
| 0 | 12498 | 70531668 | 0 |
| 1 | 12189 | 68727978 | 0 |

計139259646個の積を実際に省いたが、全scoreはほぼ変わらず、CPUよりpaired時間比約2.37/1.80倍遅い。position列同士のdotは全row順序和のままで、最大依存長は減らない。省略積数からcritical pathの短縮を計算できない。今回stage別時間やGPU利用率は測定しておらず、kernel/packing/driverのどれが支配的かは断定しない。全PF/RTK/FGO性能の採用試験へは進めず、既存の改善値とも加算しない。

## 実GTSAM+GPUの精度確認

run3/seed20260710/400epoch、cluster0/1各1prefixを新sparse GPUで実行。元D236 GPU基準に対し計400保存PF state行、全trajectory配列dtype/shape/bytes、全cfg/従来caller metadataが一致。各197valid/3skip、394nonempty GPU score、scalar fallback0。実走入力は旧GPU traceに従い、上のCPU trace corpusと同一のH数だとは仮定しない。source/input/外部tc/cssrlib/3CPU拡張はhash固定。scorerのkernelを明示選択し、元field/base bindingはproxy内で差し替える。元field自身のruntime adapter install/restoreは踏襲する。counter/storageのあるfield実走は精度確認のみで、call_secondsを性能根拠に使わない。

元NumPy対nativeの既存微小差と保存state codecのscalar NaN payload正規化を継承し、新しい許容差はない。GPU全route/multiseed/既定採用gateは未完了。今回新しくGPUへ移した数値領域はなく、既存尤度内の演算削減である。GTSAM solver/整数探索はCPUのまま。元fieldのRBPF_CUDA=0は整数探索側で、mode=gpuの尤度scorerとは別。

capacity512のstagingはhost/device各1143352B/contextで元と同じ。5timing contextsでhost/device各5716760B。後段counterは2contextsでhost/device各32768Bを追加し、shared scratchは新kernelで+552B/block。raw normal試験の最大128行列の一時device allocation計2033152Bは通常scoreに存在しない。driver管理領域/CPU heap peakは未測定。36製品source、5installed PEs、PF build PE、元数値DLL/3拡張、60既報、29archive/4555filesを保全する。

[結果](../benchmarks/rbpf_sparse_normal_20261017.json)、[完了照合](../benchmarks/rbpf_sparse_normal_completion_check_20261017.json)、[scorer](../benchmarks/rbpf_sparse_normal.py)、[probe/API](../benchmarks/probes/rbpf_sparse_normal/README.md)、[実入力比較](../benchmarks/bench_rbpf_sparse_normal.py)。
