# RBPF GPU初回行列・LUの厳密共有 (D267, 2026-10-17)

全scoreのGPU同backend component 0/2、CPU比を含むjoint 0/2、対照較正 2/2。新しいCUDA実装を明示実験として保存し、製品/既定を維持する。D265/D266の属性/確率/重み一括化を併用せず、独立したGPU数値仕事量の削減を調べた。

## 共有できる理由と保持した計算

初回の `prior + Aw.T @ Aw` は同一batchのrows/meta/pairs/config、有限DDの採用pattern、事前共分散9個のbinary64 bitsだけで決まる。候補整数の値・位置centreは右辺へ入る。全128bitのfinite-DD patternと全9prior bitsを比較し、8group/11次元だけ、最初に見つかった同じkeyを所有者にする。signed zero/1ULP差も分離する。全観測/候補/particle/候補順と任意精度整数→元FP64 DD展開は保持し、値の近似や間引きはない。

所有者の初回normal/LU/pivot/logdetを元のregister solverで生成するkernelと、各候補の元likelihood kernelの2段にした。各候補のinitialize/prepare_row/事前逆行列・右辺の元行順内積を実行し、共有LUから元と同じpivot順・前進/後退代入を行う。元solverの求解tail全文をそのまま使う。2回のIRLS・再weight・後続normal/LU/solve・最終score/135診断/status/fallbackは元のまま。非一致/所有者失敗/元finite・2^500 guard外は元solverへ戻す。初回行列の共有でも数値許容差や対称化/jitter/FP32/FMAを追加しない。

key表とownerはcall内で再作成し、数値templateを次callへ流用しない。owner配列を既存input copyへ追加する。通常scoreのoutput転送量は元のまま。usage flagの読戻しは別の精度検査APIだけで、timed scoreは呼ばない。[CUDAのstream順序](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)に従い、両kernelを同じdefault streamへ投入してから元の完了readbackを行う。独立block間のspin-waitや浮動小数point atomic/reduction順序変更はない。

## 初版の競合と別path修正

初版は110数値testsとmemcheck36を通過したが、Racecheckで2hazard群（error1/warning1）を検出したため不採用。shared `reuse` のLDS 0x6ec0とSTS 0x9490が同じ共有address 0x78b8へ重なった。source上のlane0 guardだけでは生成されたread/writeを同期できなかった。

初版source/PE/plan/log/機械語/負例を固定して保存した。別checked pathで、初期化後は書き換えないeligibilityと、normal/RHS準備後にlane0が書いて次barrier後に読む採否flagへ分離した。5個の明示置換を戻すと初版と一致し、数学式/試験/数値gateは変更していない。初版での速度測定は0回、完了した速度の再試行もない。

## 数値・同期・メモリ検査

修正版110tests PASS、既知pytest設定warning1。元/同DLL off/onの同backend score/全135診断/status/input bytesを照合し、実保存394batchを全件使用した。共有するfull support、centre差、prior1ULP/signed zero、support差、NaN/Inf/巨大・signed-zero RHS、matrix overflow/特異・微小prior、nb65/128・空行、次call・返却array所有権・closeも検査した。GPU共有の実行flag正も確認。scalar fallback/警告/strict policyを維持する。GPU完全PFでは3seed×4構成×4epochを元/off/onで進め、保存state/RNG/graph eventがbits一致した。これはfixtureのextract/search/graph solve境界内である。

修正版のCUDA memcheck/racecheck/synccheckは各36tests PASS、error/hazard/leak0。初版の失敗件数と修正版のPASSを合算しない。fault injection/全Python heap/全routeの保証ではない。CPU bodyと元numeric prefix/regular GPU kernels/register header/build configはliteral。旧3kernelのnormalized SASSは同一（regular register18336、original5144、raw probe13136命令）。新しい2kernelは別に保存する。静的命令数や削減候補数を速度・GPU化率へ換算しない。

## 実入力全scoreの速度

Tokyo run3/seed20260710の全197入力×cluster0/1。CPU、旧GPU、同DLL off/on GPU、GPU A/A、CPU A/A、on A/Aの7mode、5round×3group、41370timed/2758warm。3所有workspaceを使い、CPU/GPUは直列。許可CPU1本/BLAS1/normal priority/非占有。復元/copy/準備/hash/検証/serializationは外、元dedup/DD/cluster/packing、追加host key分類/owner upload、template生成とlikelihood両launch、完了待ち/readback/scatter/fallbackを全てtimer内へ含めた。イベント・usage query・診断読戻しをtimed callへ追加しない。

各groupの197batch integer-ns合計からpaired短縮率→3group中央値→5round中央値。GPU同backendは旧GPUと同DLL offの両方へ10%以上/4of5改善round。旧GPU A/A・CPU A/A・on A/A・off対旧GPUの全4対照abs中央値5%未満を要求する。jointは旧CPU比にも同じ10%/4of5条件を追加。全負値/対照失敗/揺れを保存し、丸め前で判定した。

| cluster | 対旧GPU | 対off GPU | 対CPU | GPU A/A | CPU A/A | on A/A | off対旧 | GPU同backend/joint |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| c0 | -13.761% | -12.034% | -109.692% | -0.447% | -1.400% | +0.113% | -0.792% | False/False |
| c1 | -10.498% | -10.543% | -76.960% | -0.342% | +1.158% | -0.234% | +0.120% | False/False |

独立auditで全41370記録の順序/件数/ns/返却hash、2758warm全結果/診断/metadataをcodecから再照合し、全対比とgateを再計算した。warm GPUは凍結済み旧GPU診断hashとも一致する。CPU/GPU間に既存の丸め差はあり、今回も同backend比較へ新しい許容差を足していない。Python scalar NaN codecの既存制限を継承し、raw native試験はarray bytesを直接比較する。

## 共有の実行と範囲

保存入力の事前構造調査では、初回11次元の同key重複がc0 2301/c1 2197件。次表はwarmのon GPU各197callから取得した実行flagで、timingから分離した精度確認。accepted pathは所有者自身も含むので、削減LU分解回数と同一視しない。右辺が不安全ならmatrixを読み込んでもLUの求解pathは元へ戻る。

| cluster | call | unique | key重複 | group member | accepted initial path | 2launch call | owner upload bytes合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| c0 | 197 | 4166 | 2301 | 2570 | 2570 | 185 | 16664 |
| c1 | 197 | 4063 | 2197 | 2471 | 2471 | 184 | 16252 |

全GPU component通過条件が0だったため、事前方針どおり追加の完全PF速度screenと実データ再走は実施しなかった。110試験内の複数epoch PF一致と実394score入力の一致を、実走の測位精度やPF/FGO速度の証明へ読み替えない。

## メモリと保全

capacity512の追加deviceはtemplate/usage flags1021952bytes＋input owner2048bytes＝1024000bytes、追加hostはkey49152bytes＋owner2048bytes＝51200bytes（workspace object/allocator overheadを除く）。off/onとも構築時に確保し、destroyで解放する。旧host/device各1143352bytesに加算する。1template1992bytes/key96bytes/owner4bytesはAPIと試験で確認。通常call内の追加allocate/freeは0。Python/NumPy heap peakは未測定。

新likelihoodの静的shared30912bytes、template kernel30908bytes、各64register/stack32bytes/spill0。旧regular shared30904bytes/64registerは維持。資源値はcompilerの静的報告で、動的占有率・純GPU時間・全FGO割合ではない。追加GPU転送はowner4×unique bytes/call、追加出力転送は0。独立GPU/contextへの通常API以外の外部writeや同一workspace並行呼出しは元の対象外。

36製品source・5installed PE・PF build PE・94過去reports・46archives8202filesを保全し、初版と修正版のsource/PE/log/検査・全raw/warm・独立audit・旧manifestを新archiveへ保存する。Ruff/compile/UTF8/EOL/diff/公開機種名/artifact policyを確認する。製品採用・既定変更・全FGO速度・全route/multiseed精度・新GPU化率の主張はない。

[結果](../benchmarks/rbpf_initial_share_20261017.json)、[完了照合](../benchmarks/rbpf_initial_share_completion_check_20261017.json)、[scorer](../benchmarks/rbpf_initial_share.py)、[修正版CUDA](../benchmarks/probes/rbpf_initial_share_checked/probe.cu)。
