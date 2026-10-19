# D252: 11×11 LU分解・求解の固定サイズ展開

2026-10-17。D251のblock内サイクル診断で約74%を占めた正規行列のLU分解・求解を対象に、実入力で多い11×11のループ境界・添字をcompile-time定数にした。元のFP64演算と順序を保ち、コンパイラでループを展開する。旧GPUに対する全score短縮はcluster0 +7.384%、cluster1 +4.239%。GPU同士のcomponent gateは0/2、CPUを含めたjoint gateは0/2。製品・既定は変更せず、全PF/FGOの速度採用試験は未実施。

## 対象と数値

凍結したD251全394warm入力のunique hypothesisを調べると、cluster0は4166件中3370件、cluster1は4063件中3315件がdimension11だった（計6685/8229、約81.2%）。このサイズだけを特殊化し、3〜10は元のsolve_normalへ戻す。DDの有限性によるkept row/group数も元の定義のまま。実GPU実走の候補数がCPU捕捉corpusと同じだとは仮定しない。

partial-pivotの検索順、同値時の選択、row swap、logdet和、除算、LU更新のj/i/k順、pivot置換、前進・後退代入とstatusを維持。full asymmetric prior、全候補・観測、2回のIRLS更新、閾値とRNGも維持。対称化/jitter、近似逆数、FMA、fast mathは加えない。元numeric prefix/CPU body/通常GPU kernelはliteral copy。同DLL元kernelの5144 normalized SASS命令/hashは旧DLLと一致する。

新kernelは10464命令、96register、shared30904B、stack32B、spill0。元は56registerでshared/stackは同じ。ループの展開で命令数は増えるため、展開率や演算数を速度利得と扱わない。新kernel内のgeneric-controlは同じcode/resourcesで特殊化を無効にする。旧kernelとはresourceが違うため、その旧GPU比は等速対照の条件にしない。元D226のpageable/synchronous転送を維持し、D248/D249/D250は組み合わせない。

## 差分検証

198tests PASS。3seed×3size×6optionの同backend全score/normal/solution/logdet/status、row0/1/63/127、元のfallback/例外/警告、24context lifetime/返却array所有権、3seed×4PF設定×4epochの全保存state/RNG/graph eventを比較した。

独立したraw solve APIでは3744行列（dimension3〜11×13種類×32）を比較。SPD、非対称、row pivot、同値pivot、特異、重複row、負のdet、NaN/Inf、overflow、最小subnormal、signed-zero、桁差を含む。11×11特殊化を通る416行列を含め、LU/pivot/解/logdet/statusのdtype/shape/bytesが一致した。raw probeは通常scoreから呼ばない。raw probe kernelは7744命令、104register/shared30904B/stack0/spill0。

CUDA memcheck/racecheck/synccheckは各16対象subset PASS、error/hazard/leak0。48個の追加固有testとは数えない。各suiteに既存pytest asyncio設定warning1件。allocation/非同期障害の注入試験は未実施。実行したsource/DLLは凍結し、数値許容差を追加しない。元NumPy/native間の既存丸め差と保存state codecのscalar NaN payload正規化は継承する。

## 実入力の全score時間

全394入力、8mode×5round×3group、47280timed/3152warm。最適化CPU、旧GPU、同DLL元kernel、新generic、新fixed、旧GPU/CPU/fixed A/Aを順序rotate/reverseで比較。全warmの同backend score/135diagnostic/status/input bitsと全timed結果hashが一致。GPU warmは以前の凍結GPU基準とも照合した。

workspace/warm/復元copy/hashはtimer外。dedup/DD/cluster/pack/guard、fixed11 dispatch、全転送/kernel/完了同期/返却、fallback/scatterはtimer内。allowed CPU1本/BLAS1/normal priority/非exclusive、GPU処理は直列。全197batchの整数ns合計→round内3group paired率median→5round median。事前gateは旧GPU・同DLL元GPU・新generic各10%以上、各4/5round以上正。GPU/CPU/fixed A/Aと同DLL元対照のabsmedian<5%。CPU比も10%以上をjoint gateとし、別に報告する。全rawを保持し、完了した候補を再測定しない。

時間列は197batch合計median÷197で、個別batchのp50ではない。短縮率は別のpaired統計量。

| cluster | CPU ms | 旧GPU ms | 新generic ms | 新fixed ms | 対旧GPU短縮 | 対generic短縮 | 対CPU短縮 |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.241 | 2.128 | 2.147 | 2.013 | +7.384% | +5.660% | +11.665% |
| 1 | 3.265 | 3.736 | 3.628 | 3.497 | +4.239% | +3.138% | -0.211% |

| cluster | GPU A/A | CPU A/A | fixed A/A | 同DLL元対照 | controls |
|:---:|---:|---:|---:|---:|:---:|
| 0 | -0.243% | +0.274% | -1.376% | +0.595% | True |
| 1 | -0.739% | +0.719% | +0.922% | +1.108% | True |

事前に定めたmedian対照条件は両設定で通ったが、cluster0の第2round GPU A/Aは-11.41%、CPU比は第1round -62.53%から後続round +10〜14%へ振れた。全round/rawを保持し、単発のCPU優位や過去実験との絶対時間差を一般化しない。今回の10%基準未達の判定を維持する。

GPU同士の短縮と、最適化CPUからGPUへ切り替える利得を区別する。全RTK/PF/FGOの高速化、GPU利用率やGPU化率には換算しない。GTSAM solver/整数探索はCPUのままで、新たな数値offload領域はない。

## 実GTSAM+GPUの精度

run3/seed20260710/400epoch、cluster0/1各1prefixをfixed GPUで実行。元D236 GPU基準に対し計400保存PF state、全trajectory配列のdtype/shape/bytes、全cfgと従来caller metadataが一致。各197valid/3skip、計394nonempty GPU score、scalar fallback0。fieldのみ全diagnosticを取得してdimension11を両設定で確認し、呼出元へは元のmetadataだけを返す。追加のscore callはなく、diagnostic readback/counter付きfield時間は性能の根拠に使わない。全route/multiseed精度gateとGPU既定採用は未完了。

stagingはhost/device各1143352B/context、timing5contextsで各5716760Bと元と同じ。通常scoreに追加counter/転送はない。raw検査最大256行列のdevice一時確保は863232Bで、通常scoreにはない。driver管理領域/CPU heap peakは未測定。36製品source、5installed PEs、PF build PE、元DLL/拡張、64既報、31archives/4890filesを保全する。

[結果](../benchmarks/rbpf_fixed_lu_20261017.json)、[完了照合](../benchmarks/rbpf_fixed_lu_completion_check_20261017.json)、[scorer](../benchmarks/rbpf_fixed_lu.py)、[全score比較](../benchmarks/bench_rbpf_fixed_lu.py)、[probe/API](../benchmarks/probes/rbpf_fixed_lu/README.md)。
