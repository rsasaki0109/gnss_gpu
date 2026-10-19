# D287: 同期転送のhost stagingをページ固定化 (2026-10-18)

同じ修正済みregister-LU数値kernelを使い、host stagingだけを通常heapからcudaMallocHostへ替える候補を実装した。通常の全score screenは0/2、15秒idle後のscreenはFalse。37unique testsと12対象memcheckを通過し、実RTKの400状態・2最終状態・2軌跡全114配列は従来と一致した。RTK速度・製品/既定への採用は未確立。

通常条件の対old短縮はcluster0 -4.372%、cluster1 -2.154%。改善roundはそれぞれ0/5、0/5で、通常対照の両clusterが合格した条件で候補は遅くなった。idleではold A/Aのabs5%未満が2/5、同DLL pageable対照が1/5に留まり、必要な各4/5を満たさない。候補の好都合なidle中央値だけを高速化として採用しない。この同期pinned候補は不採用として保存し、同候補の完了測定を繰り返して合格を探さない。

## この候補を試す根拠と範囲

D286は通常RTKの先行native1112ms中H2D包含区間1074ms、別の15秒idle対照でも903ms中868msを観測した。転送単体時間やdriver原因とは断定しなかった。旧D226/D249のpinned比較は別kernel/条件の完了済み結果として保持する。今回はD286の同じregister kernelと新しいidle再発条件を対象に、同期API/既定streamを保ったままallocation方式一つを比較する。

NVIDIAの[CUDA12.8.1 API同期仕様](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/api-sync-behavior.html)は、通常host memoryからの同期H2Dでは開始前のstream同期と内部stagingがあり、pinned H2Dもhost同期、D2Hはどちらもcopy完了後に返ると説明する。一般にAPIはresource競合等でも待ち得るため、仕様から約1秒の原因や候補の高速化を推定しない。[Page-locked memory仕様](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#page-locked-host-memory)も参照する。

## 数値処理を保持した実装

D286全sourceに対する変更はWorkspaceの選択/ownershipフラグと対応destructor、明示setup/info/検証APIだけ。2つのWorkspace編集を戻してAPI追記を除くと全文がD286へbyte一致する。rbpf_scoreとrbpf_score_traceの全関数、CPU/device数値関数、register header、2つのcudaMemcpy/既定stream/launch順はそのまま。新旧DLL全3device関数のSASS命令/operand/encoding wordsとresourceが完全一致した。

作成直後・score前に一度だけpageable/pinnedを選ぶ。同じDLLのpageable対照を残す。pinnedの両allocationが成功した後に既存heap buffersを解放して置き換え、部分失敗時は取得済みpinnedだけを解放し、元buffersのownershipを残してPython constructorがclose/例外に進む。黙った速度fallbackはない。選択済みの再選択は拒否する。raw ABIにはscore前/単独ownerというpreconditionがあり、Python wrapperはconstructor内で選択する。allocation/device failure注入は未実施。

通常の速度比較は元RegisterLUScorer.scoreの同じcode objectを使い、Python/nativeの時計markerを呼ばない。明示診断用classだけがD286 traceを継承する。setup時のpointer attribute検証は性能timer外で、両pinned bufferがCUDA host memory型と一致することをtestで確認した。新stream/event/同期API/keepalive/power変更はない。

capacity512のhost payloadは1,143,352B/context、device payloadも同量で元と同じ。候補はhostをpage-lockし、新device bufferは加えない。constructor中は初期heapと新pinnedを一時的に両方持ち、成功後heapを解放する。runtime/driver/stack、Python trace/packとheap peakはこのpayload数値に含めない。

## 通常の全score screen

両cluster197入力を全て使用し、old/new-pageable/new-pinned/old-AAの4方式を5round×3groupで巡回/反転、計23640timed・1576warm。全返却digestを保存済み旧GPU基準へ照合し、warm入力の不変性も確認した。copy/restore/hash/IO/setupはtimer外、元scoreのdedup/DD/cluster/packing/H2D/計算/完了待ち/D2H/scatter/fallbackは全てtimer内。CPU1/BLAS1/通常priority、host/GPUは非占有。

判定はgroup内の全197件の整数ns合計からpaired短縮率を求め、round内3group中央値、5round中央値。候補は旧GPUと同DLL pageableの両方へ10%以上、4/5round以上改善が必要。old A/Aと同DLL pageable対照は両方abs中央値5%未満、かつ各4/5round以上abs5%未満が必要。負値・遅い入力・失敗controlを全て保持し、再試行しない。下のms列はcorpus合計中央値/197であり、個別callのp50ではない。

| cluster | old ms/batch | new pageable | new pinned | pinned対old paired短縮 | pinned対pageable短縮 | old A/A | screen |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.3392 | 3.3040 | 3.4828 | -4.372% | -4.460% | +0.409% | False |
| 1 | 4.5355 | 4.4790 | 4.6090 | -2.154% | -3.446% | +1.538% | False |

全5round・両controlの4/5条件は結果JSONへ保存する。同じGPU backend内でtransportを比較するscreenであり、最適化CPUとの比較や従来のGPU共同採用gateは新たに実施していない。whole PF/FGO/連続RTKの速度やGPU使用率へ外挿しない。

## 各callの前に15秒待つ対照

同じfield Python3.12.10/NumPy2.4.6と先頭実入力、同じ4方式を使う。warm後、各方式の各call直前に個別に15秒待つ。4方式×5巡=20callを巡回/反転し、次の方式が前のGPU起床だけを受け取る比較を避ける。待ち時間はscore時間に含めず、実待機nsを全件保存する。通常screenと同じ10%/4of5と両control5%/4of5を事前に宣言した。

| 巡 | old全score ms | new pageable | new pinned | old AA |
|---|---:|---:|---:|---:|
| 1 | 745.774 | 36.666 | 2.380 | 2.941 |
| 2 | 1.621 | 4.847 | 2.988 | 6.326 |
| 3 | 2.063 | 4.927 | 1.733 | 2.569 |
| 4 | 778.903 | 3.718 | 4.596 | 789.696 |
| 5 | 757.875 | 749.992 | 745.817 | 770.176 |

paired短縮は対old +15.985%、対同DLL pageable +38.365%。control pass=False、screen=False。1入力/20callの局所latency試験であり、長期頻度やRTK全体の改善率とは扱わない。4workspaceを同時に保持しており、D286の1workspaceとは別条件である。pinned allocationの生存が他方式のruntime/driver状態へ影響しないことも証明していない。この試験だけで単独contextの起床原因や改善を決めず、同時に保持した今回の4方式内の比較に限定する。通常screenとidle screenを混ぜたり、どちらかの失敗を好都合な時刻で置換しない。

全速度試験の後に、両新方式で初回/15秒後/直後の3callずつをhost marker付きで別途診断した。この6callは速度gateへ使わない。

| 診断call | native host ms | H2D包含区間 ms | launch包含区間 ms | D2H包含区間 ms |
|---|---:|---:|---:|---:|
| pageable / initial | 2.3020 | 0.0662 | 0.0295 | 2.1373 |
| pageable / after_idle15 | 775.3652 | 739.3276 | 0.1975 | 35.7078 |
| pageable / immediate | 2.0860 | 0.0551 | 0.0272 | 1.9371 |
| pinned / initial | 2.2355 | 0.1703 | 0.0469 | 1.9511 |
| pinned / after_idle15 | 761.2875 | 757.5568 | 0.7363 | 2.9217 |
| pinned / immediate | 1.9838 | 0.0775 | 0.0165 | 1.8450 |

host区間はruntime処理/待機/host命令/割込/観測費用を含み、純粋なデータ転送やGPU kernel時間ではない。D286と同じ生QPC/user/kernel標本とgenerationを全て保持し、整数保存則とnative<=Python native<=scoreを有理数で照合した。

## 実RTK精度と検証

候補pinnedの独立2driverを元の先行反転protocolで全400epochずつ実行し、通常GC/モデル/seed/構成を維持した。全800RTK/400PF/394native traceを保存。先行nativeは1232.8466ms、うちH2D包含区間1229.2020ms、直後の後行nativeは2.9668msだった。pinnedだけを使うこの実RTKでも長いH2D区間が残り、ページ固定化だけで待ちを取り除ける根拠は得られなかった。原因をdriver/powerへ断定せず、別時刻のD286の絶対時間から改善率も作らない。

両armの200postprocess状態は全体でA/A完全一致、従来PF-return基準とは事前宣言済みroot rtkだけを除き一致する。両最終状態はその境界も含め旧post-driver goldへ一致し、軌跡2本の全114配列はdtype/shape/bytes一致した。観測付き一組をRTK速度の採否に使わない。full route/多seed精度を追加で確立したとは言わない。

37testsは明示選択/不正値/immutable setup/pinned pointer型/情報buffer容量、6構成のCPU/GPU元・両新方式・trace有無の数値/diagnostic bits、両cluster全394実入力と入力不変、0/1/63/127rowsとshape変更/返却配列所有、9fallback/警告/例外条件、24allocation lifetime/double close、負round/controlを残したgateを確認する。CUDA memcheckはこのうち12対象subsetを再実行し、error/leak0。12件をunique37件へ加算しない。device code完全一致のため新しいrace/sync再実行は行わない。

独立auditorはhelperをimportせず、元source全文復元・SASS/resource・全23640+20timed順序/digest/整数ns/paired統計・394+6trace・400状態/2最終状態/114配列を再検算した。最初のauditor生成は改行の構文errorを実データ読込み前のcompileで検出。初版generator/auditorと失敗記録を残し、別のchecked版を使用した。数値source/PE/速度code/gateの修正や測定取り直しはない。

134既報・66archives/15653files・36製品source・5installed PE・PF buildを保全する。D282速度FAIL/D284校正FAILと全phase2CPU差は維持し、epoch除外・別gate再判定はない。新しい製品/既定変更や精度犠牲はなく、候補は検証用APIとして保存する。

[結果](../benchmarks/rbpf_pinned_transport_20261018.json)、[完了照合](../benchmarks/rbpf_pinned_transport_completion_check_20261018.json)、[scorer](../benchmarks/rbpf_pinned_transport.py)、[screen](../benchmarks/bench_rbpf_pinned_transport.py)、[probe](../benchmarks/probes/rbpf_pinned_transport/README.md)、[tests](../tests/test_rbpf_pinned_transport.py)。
