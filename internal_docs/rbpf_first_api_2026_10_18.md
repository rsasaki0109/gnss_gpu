# D288: 待機後の最初のCUDA操作を分離 (2026-10-18)

ページ固定化でもH2D包含区間の長い待ちが残ったD287を受け、none/照会/同期/H2D1B/H2D64KiB/memset1Bを各3回、計18個の別process・各1workspaceで観測した。計算kernelを追加せず、前処理API・完了待ち・その後の完全score・両方の合計を保存した。スコアだけが短くなっても、待ちを前へ移したものを高速化とは呼ばない。

H2D1BのAPI中央値は727.556ms、H2D64KiBは753.643ms、device memset1Bも717.739msだった。一方、照会/同期API自体は各0.0323/0.0434msで、その後のscore側に待ちが残る。小さなdevice操作を先に行うとfollowing scoreは約3msへ短くなるが、probe+score合計の中央値は751.432〜790.575ms。コピー量の削減や計算の省略だけで、この長い待ちが解消すると見込む根拠は得られなかった。driver/powerの根因までは特定していない。

48unique tests PASS、memcheck9対象subsetはerror/leak0。保存済み実入力394件のそれぞれで全6操作後にfresh scoreを実行し、計2364比較の値/全diagnostic/statusが元GPUとbit一致した。18試行の54score返却も旧GPU参照と一致する。今回は新しいRTK軌跡試験を行わず、数値TU/両score全関数/既存integrationは保持した。

## 条件と変更範囲

D286の全28315byte translation unitとheaderをそのままprefixに置き、別のprepare/probe APIだけを追記した。新旧DLLの全3device関数はSASS/opcode/operand/encoding wordsとresource完全一致。旧score・元FP64/IRLS/prior/観測/仮説/閾値/順序を変えない。

prepareはCPUだけでprivate host stagingをzero初期化し、15秒idleより前に終える。probeはnoneならCUDA API0回、cudaStreamQuery/default-streamとcudaDeviceSynchronizeなら各1回、H2D1B/H2D65536B/cudaMemset1Bなら実行後のdevice synchronizeを含め各2回。memsetはruntime内部のdevice作業を使う場合があるが、推定kernelは新設しない。probeはscoreのTLS traceを上書きしない。

H2D/memsetはprivate native入力stagingを変える。次の元scoreは元通りPythonから全active inputをpack/uploadしてから計算する。元入力/graph/particle/RNGへ診断値を入れず、前epochの尤度を使い回さない。単独owner/完了済みworkspaceでのみprobeする。guard対象外のraw pointer、同時呼出し、CUDA障害注入は対象外。

NVIDIAの[API同期仕様](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/api-sync-behavior.html)に従い、device memsetやpageable H2DのAPI返却だけをGPU完了とは扱わず、probe内の完了待ちも計測する。[stream query](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__STREAM.html)と[device synchronize](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__DEVICE.html)も参照した。APIの壁時計はruntime処理/待機/割込/host観測費用を含み、純粋なGPU計算や転送時間ではない。

## 新process・1workspaceの観測

元実入力cluster0/index0を使い、各試行で新しいfield Python3.12.10/NumPy2.4.6 process、1つのpageable workspace、同じ数値DLLだけを用意する。CPUのcanonical/parent拡張は元のまま。warm score1回→CPU prepare→15秒idle→probe+完全score→直後のscoreを全て保存し、返却/入力digestを確認する。18childを順番に起動し、前childの正常終了後に次へ進む。

6方式を3巡で巡回/反転し、全試行を残す。D287の同時4workspaceと比べ条件は異なり、絶対時間差から方式の改善率を作らない。CPU1/BLAS1、normal priority、host/GPUは非占有。clock/power設定・keepalive・外部profilerは変更/導入しない。新しい速度gateも作らず、D282/D284/D287のFAILを保持する。

各列は3試行の個別中央値。API中央値と完了待ち中央値を足して、他列や元計測の排他的内訳へ読み替えない。合計は各試行で実際に測ったprobe呼出し全体とfollowing scoreを含むwallで、同じ試行内の整数nsとして足し戻せる。

| 最初の操作 | API host ms | 完了待ち host ms | 続く全score ms | probe+score合計 ms | 直後score ms |
|---|---:|---:|---:|---:|---:|
| none | 0.0019 | 0.0008 | 768.1045 | 768.2303 | 3.2697 |
| query | 0.0323 | 0.0009 | 752.3283 | 752.4455 | 3.0924 |
| sync | 0.0434 | 0.0010 | 751.2745 | 751.4321 | 2.6758 |
| h2d1 | 727.5559 | 32.7382 | 3.6340 | 764.3622 | 3.7088 |
| h2d64k | 753.6429 | 33.7299 | 3.4262 | 790.5754 | 3.1838 |
| memset1 | 717.7387 | 33.4775 | 3.4579 | 754.2055 | 3.1011 |

### 全18試行

| trial | 巡 | 操作 | API ms | 完了待ち ms | following score ms | 合計 ms |
|---|---:|---|---:|---:|---:|---:|
| 0 | 1 | none | 0.0024 | 0.0009 | 768.1045 | 768.2303 |
| 1 | 1 | query | 0.0283 | 0.0006 | 752.3283 | 752.4455 |
| 2 | 1 | sync | 0.0425 | 0.0013 | 753.3877 | 753.5345 |
| 3 | 1 | h2d1 | 733.3835 | 32.8865 | 3.6340 | 770.0373 |
| 4 | 1 | h2d64k | 715.7086 | 33.6798 | 3.4262 | 752.9455 |
| 5 | 1 | memset1 | 717.7387 | 33.2864 | 3.0594 | 754.2055 |
| 6 | 2 | none | 0.0016 | 0.0006 | 755.0894 | 755.1851 |
| 7 | 2 | memset1 | 703.6259 | 33.4775 | 3.4579 | 740.6778 |
| 8 | 2 | h2d64k | 817.9951 | 39.1061 | 4.3480 | 861.5875 |
| 9 | 2 | h2d1 | 0.0556 | 0.8967 | 2.7856 | 3.8235 |
| 10 | 2 | sync | 0.0434 | 0.0010 | 17.6739 | 17.8349 |
| 11 | 2 | query | 0.0330 | 0.0010 | 730.2506 | 730.4053 |
| 12 | 3 | sync | 0.0573 | 0.0008 | 751.2745 | 751.4321 |
| 13 | 3 | h2d1 | 727.5559 | 32.7382 | 3.9226 | 764.3622 |
| 14 | 3 | h2d64k | 753.6429 | 33.7299 | 3.0728 | 790.5754 |
| 15 | 3 | memset1 | 737.4241 | 36.6601 | 3.6390 | 777.8415 |
| 16 | 3 | none | 0.0019 | 0.0008 | 792.6478 | 792.7702 |
| 17 | 3 | query | 0.0323 | 0.0009 | 754.6429 | 754.7784 |

3回ずつの局所診断であり、一般の起床頻度/長期throughput/driverまたはpowerの原因までは決めない。前処理で待つかscore側で待つかを観測し、合計時間を無視して新しい最適化へ採用しない。小さな転送量で長く待つ例があっても、hardware/OS全体の因果を一意に同定したとは扱わない。

## 精度・観測の検証

48件はprobe header/type/長さ/operation bytes/API count/phase順/時計単調性/生整数保存、6操作後の元CPU/GPU値とdiagnostic bits、両cluster全394入力と入力不変、score TLS traceの非上書き、raw不正値/容量guard、prepare範囲/小workspace拒否、結果array所有権/closeを検証する。394×6ループをunique testsへ加算しない。CUDA memcheckはこのうち9対象subset、error/leak0。

独立auditはhelperをimportせず、source全prefix/SASS/resource、全18processのplan/argv/PID/正常終了/保存hash、18probe/54scoreの全生QPC/CPU/phaseと有理数包含、probe+scoreの整数timeline和、旧参照digest、表の全中央値を再検算した。QPCを丸めてから足さず、CPU時計の0/量子化をclipしない。profilerや追加hash/保存は測定区間へ挟まない。

最初のauditは起動用PIDと数値worker PIDを同一と仮定して停止した。保存済み全18件ではworkerのparent_pidが起動用PIDに一致し、PIDが別になっていた。起動用processと数値workerの親子関係を照合する別checked auditorを使い、初版audit/plan/失敗記録を保全した。18というprocess数は数値worker数で、起動用processを含むOS総数ではない。数値source/test/PE/trialを変更・再実行していない。

数値modelを変更しない診断のため、以前完了したフルRTK試験を再実行しない。新しい測位精度や全route/多seed/全FGO速度は主張せず、2364実score復元比較と54同一入力返却を今回の精度範囲として明示する。136既報・67archives/16135files・36製品source・5installed PE・PF buildを保全し、製品/既定/許容差は維持する。

[結果](../benchmarks/rbpf_first_api_20261018.json)、[完了照合](../benchmarks/rbpf_first_api_completion_check_20261018.json)、[scorer](../benchmarks/rbpf_first_api.py)、[1trial caller](../benchmarks/bench_rbpf_first_api.py)、[probe](../benchmarks/probes/rbpf_first_api/README.md)、[tests](../tests/test_rbpf_first_api.py)。
