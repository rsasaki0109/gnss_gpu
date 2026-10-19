# D258: 観測行数に合わせたGPU shared scratch (2026-10-17)

全入力m<=60は60行用、61..64は64行用、65..128は従来の同期修正版128行用kernelへdispatchする。観測を除外した後のs.countでは選ばない。全観測・仮説・パーティクル・IRLS・FP64・非対称prior・演算順・閾値・RNGを保ち、確保するrow配列の容量だけを変える。製品・既定を変更しない独立DLL実験。

## 全scoreの結果

両条件のGPU component基準を満たしておらず、本変更の高速化採用は見送る。 同期修正版比+1.830/-1.051%。GPU component gateは0/2、最適化CPUとのjoint gateは0/2。全PF/FGOの高速化や既定への採用は確立していない。

以下の正値は時間短縮、負値は遅化。

| 対比 | cluster0 | cluster1 |
|---|---:|---:|
| compact_vs_synced_gpu | +1.830% | -1.051% |
| compact_vs_same_DLL | +1.840% | -0.208% |
| compact_vs_full_capacity | +0.613% | +0.233% |
| compact_vs_cpu | +12.564% | +2.824% |
| cpu_AA | +1.218% | +1.036% |
| synced_AA | +1.429% | -0.117% |
| compact_AA | -0.968% | +1.167% |
| original_control | +0.783% | -1.568% |
| full_capacity_control | +1.265% | -2.063% |

394実走保存score入力（各条件197）×8modes×5round×3groupで47280timed、3152warm。各groupの197件integer-ns合計を対比し、round内3group中央値、その5round中央値を使う。基準は凍結同期修正版・同DLL内のliteral同期修正版・新templated128対照の各々に10%以上、4/5round以上正。CPU/同期修正版/候補AAと同DLL literal対照は各absolute median<5%。CPU jointは加えてCPU比10%以上・4/5round正。全roundとrawを保存し、完成した計測をやり直していない。

時計には従来scoreのdedup/DD/cluster/packing/upload/容量dispatch/kernel/完了待ち/readback/scatter/fallbackを含み、入力復元・copy/setup・warm・hashは外。timedにはdiagnostic counter/eventを入れない。許可CPU1個、BLAS1、通常優先度、非占有環境。別実験の絶対時間と比較しない。

対照の合否は事前指定の5round中央値に対する基準で、各roundがすべて5%以内という意味ではない。c1の同期修正版AAは1roundで−11.558%だった。CPU対比も初roundでは−68.633/−50.801%、後続roundでは正に転じ、条件間の変動が大きい。これらを除外・再計測せず保存し、CPUに対する安定した優位や既定採用の根拠とはしていない。

## 資源と入力規模

| kernel | shared bytes | registers/thread | API予測の最大active blocks/SM |
|---|---:|---:|---:|
| literal128 | 30904 | 64 | 2 |
| compact60 | 16216 | 64 | 4 |
| compact64 | 17080 | 64 | 3 |
| templated128 | 30904 | 64 | 2 |

cudaFuncGetAttributesとcudaOccupancyMaxActiveBlocksPerMultiprocessor（128threads、dynamic shared0）を自プロセスから照会した。deviceは24SM、shared上限65536B/SM。モデル名は取得・記録していない。これは容量に基づく予測であり、実際の同時実行数・使用率・速度・全PF GPU化率ではない。[CUDA occupancy API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__OCCUPANCY.html)。

同じ凍結入力のunique仮説数は両条件中央値20、範囲13..44/13..42。従来版でも予測上は全体48blocksまで収まり、全caseがこの範囲内である。sharedを減らして最大block数を増やしても、この小さなgridで有効な並行実行が増えるとは限らない。これを実行時の占有や遅化原因の証明とは扱わず、採否は上の全score比較で判断する。

全kernelは18336SASS命令・64regs・stack32B・spill0。literal同期修正版とtemplated128対照は正規化SASSも同じ。60/64版は同じ命令数でshared offsetが異なる。元の逐次kernel5144命令・raw register probe13136命令も同一。raw新probeは60/64/128各13136命令・80regs、generic2600命令・78regs、stack/spill0。静的PE検査であり実行/JITコードや動的占有ではない。

## 数値と実走精度

342tests PASS。raw7872行列（n11は2357）で汎用元LUとcompact60/64/128のLU全要素・pivot・solution・logdet・statusがビット一致。3..11次元、部分failure、asymmetric、pivot ties、NaN/Inf、極小値、backsolve overflow、signed zero、相殺、1..256混合batchを含む。59/60/61/63/64/65の境界、空/最大入力、Huber/cluster切替でも全score/135診断値/input bitsが一致。12種類の4epoch PF比較で保存状態・RNG・graph eventsも一致。新しい数値許容差はない。

CUDA memcheck/racecheck/synccheck各81subset PASS、error/hazard/leak0。各suiteに既存pytest設定warningが1件。全warmは凍結GPU参照の診断/status/scoreと入力bits、timed全件は同backend返却hashが一致し、全rawの組合せ・順序・統計を独立auditした。CPUとGPUの相互bit同一を主張するものではない。

実走run3/seed20260710/400epoch prefixをcluster0/1で再走し、各200保存PFstate・全trajectory配列（dtype/shape/bytes）・全cfg・従来caller metadataがD236同GPU参照と一致。各197有効score、3skip、scalar fallback0。fieldは診断付き精度確認で、速度比較ではない。以下の容量別呼出数はhost入力mとdispatch sourceからの推定で、直接のdevice counterは追加していない。

| cluster | 60行用 | 64行用 | literal128 |
|---|---:|---:|---:|
| 0 | 164 | 17 | 16 |
| 1 | 164 | 17 | 16 |

元のNumPy/native差・scalar NaN codecの既存限界を引き継ぐ。fullroute/multiseed精度gateやallocation/async障害注入は未実施。FGOのGTSAM solverはCPUのまま。

## 保存と制約

元の数値prefix、CPU body、original/corrected kernel、register helperをliteral保持。compactの型template・row配列extent・関数修飾だけを戻すと全数値bodyが元と一致する。status順序付け2個のsyncwarpを維持。元の実験source/helper/testは一切変更せず、今回も初回実行後は凍結している。テストplanの説明末尾にはコピー元のblock-element/serial substitution表記が残るが、本実験の解法はD253のregister-row/distributed solveであり、source照合と実テストがその範囲を示す。

scoreの追加buffer/転送/counterなし。5timing contextの従来numeric stagingはhost/device各5716760B、raw test最大temporary device1455104B。driver/module/heapピークは測定していない。36製品source・5installed PE・PF build PE・過去76reports/37archives6122filesを維持し、製品/既定変更なし。

[結果](../benchmarks/rbpf_compact_scratch_20261017.json)、[完了照合](../benchmarks/rbpf_compact_scratch_completion_check_20261017.json)、[全score driver](../benchmarks/bench_rbpf_compact_scratch.py)、[CUDA helper](../benchmarks/probes/rbpf_compact_scratch/compact_scratch.cuh)、[tests](../tests/test_rbpf_compact_scratch.py)。
