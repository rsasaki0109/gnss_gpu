# 内容と順序が同じ仮説の登録計画を再利用する (D279, 2026-10-18)

実GPU/c0の全候補を残し、登録処理206522回を9751回へ95.278%減らした。33tests PASS、200保存PF状態・全測位軌跡/cfg・RNG/feedbackはgoldと一致した。完全PF15360timed/1024warmを1回実施し、速度gateはCPU 0/2、GPU同backend 2/2、GPUjoint 0/2。新しいGPU数値kernelや製品/既定の変更はない。

## 変更と所有権

D241で検証済みの4つの可逆adapter置換だけを再利用し、登録blockを新しいPython計画へ渡す。D241の不採用C++loop、D230の確率cache、D273の個別suffix案を速度採用扱いして組み合わせない。同じ新adapterで計画をoffにした対照は、元Python登録blockを1回そのまま実行する。元weight/likelihood/choice/tailのcodeと全数値式を保持する。

全粒子のHからgroup/sat辞書順・gauge・任意精度整数を保持するordered keyを作り、完全に一致するHのprefix登録結果を呼出し内だけ再利用する。common登録は最初の粒子のprefix後に遅延生成し、元のfirst-seen key/H/pending順を保つ。元と同数の候補recordと新しい粒子別listを作る。survive/stayは各粒子自身のH、releaseは粒子ごとの新しい空dictを使う。extend/shrinkの数値計算回数と出力共有は元のMoveReuseに一致し、確定/再抽選/cluster用copyは元のままで粒子を分離する。

全候補・prior・FP64・IRLS・閾値、各粒子の元の正規化/重み式とrng.choice、FIX/feedback/graphを変えない。確率やmatrixを新たにcacheせず、計画はscoreより前のstage終了時に解放し、epoch/graph callbackを越さない。logicalな候補を落とす代わりに、既に完了した純粋な登録・検索だけを省く。

guardはexact元view/config/problem/Particle、builtin ordered integer H/list/cache、既知の数値helper/code/global/builtin/NumPy binding、元native canonical exportとprofile/traceなしを確認する。未対応型・hook・変更したhelper・特殊policyは元blockまたは元whole-stepへ戻る。CPython3.12の固定private実験であり、任意private namespace、foreign concurrent mutation、audit hook、OOMの一般保証ではない。追加heapのpeakは未測定、native/CUDA bufferと転送は増やさない。

## 処理量と精度

Tokyo run3/seed20260710/400RTK epoch、GPU/c0の新候補を1prefix実行。200PF callの元PF-return境界のJSONL全行literal、全trajectory array dtype/shape/bytes、全cfg/入力/外部source/PEをgoldへ照合した。197valid/3skip、GPUscore197/scalarfallback0/追加転送0。

全logical登録206522件は同じで、実際に行う登録9751件、再利用196771件、prefix計画2118個/再利用10490粒子、common record2561件/再利用161343件。native canonical呼出しも10394件減った。削減率は登録回数であり、速度/メモリbytes/全allocationの減少率ではない。

`move_reuse`、`move_key_memo`、`native_canonical`の3つの作業量mapは実際の呼出し減少に合わせて変わる。省略したnative呼出しを実行したと数えない。元の全logical候補数、extend/shrinkの数値呼出し数、memo request/source/identity/emptyの加算、native呼出し差の一致を検査し、残る全legacy metadataは順序を含めexact比較する。保存state/rtk keyや数値値を除外した比較ではない。既存のscalar NaN payload codec制約は継承する。

## 試験と初期失敗

本体28tests、集計gate5testsがPASS。scalar/nativeCPU/GPU×2seed×2cluster×4連続epochをoriginal/off/onで照合した。候補/first-key/pending/record順、粒子自身のH参照、fresh release/list、巨大gauge/空group/辞書順/alias、custom map/NumPy binding/canonical/ordered-key/copy hook、profile保持、元RNG配列変更/例外順とcleanup、strict元例外/skipを確認する。

初回17FAIL/9PASSを保全する。16件はguardが通常入力もdeclineしたためで、full sourceとfixture ASTの既知2種類でCPythonがLOAD_GLOBAL/LOAD_ATTRの呼出しflagを変えることが原因だった。別checked sourceは2つの凍結code familyだけを認め、6statement追加と1比較演算を戻すと元source ASTへ一致する。数値/stage bodyは変更しない。残る1件はstrict元stepのunderflowを成功扱いしたtest前提で、別testにて同じ例外type/textと全partial state/RNGを比較する。初回testはbyte不変でfixtureへ移し、元path/SHA mappingを残した。

preparerの誤prior path、inline AST undoのindentation拒否、未作成testへのpatch拒否も保存する。これらの準備失敗では速度測定を開始していない。checked実装/testは最初の数値利用から固定し、完了済み性能を再計時しない。checked testの初回CRLFもsource SHA付き例外として維持する。publisherの最初のLF規約assert拒否は公開前であり、testや性能を再実行していない。元zero priorのdivide-by-zero warning2件は元tail/offと同じである。

## 完全PFの性能screen

実走のexact検証/197valid全適用/0fallbackと50%以上登録削減という事前条件を満たした後、新候補のfirst sample setを実施した。対照の揺れを踏まえ、最初から16個の異なるRNG初期値、4連続epoch、8mode、5round×3group、2条件を固定した。c0は64粒子/6unique/64rows/4groups、c1は64粒子/6unique/128rows/8groups。original/off/onのCPU/GPUとoriginal CPU/GPU A/Aをserialでrotate/reverseする。

各epochで全8modeの入力準備→全8完全PF計時→全8状態のcopy/hash/比較/保存を行い、調製やsnapshotをmode間のtimerへ混ぜない。全guard、ordered fingerprint、計画/recordの確保・再利用、元scoreの転送/完了待ちとPF tailを含む。通常GCでcallback/forced collection/threshold変更なし。1scorer/許可CPU1本/BLAS1/normalpriority/非占有。抽出/整数探索/時刻/衛星/graph solveはfixture境界であり、実RTK/FGO全体やGPU使用率の測定ではない。

1groupは16seed×4epochの完全PF integer-ns合計。3groupのpaired短縮中央値→5round中央値10%以上かつ4/5改善roundを、D236とoffの両方に要求する。A/Aはgroup差の絶対値中央値→5round中央値5%未満かつ4/5roundが5%未満。正負相殺で通さない。GPUjointはcandidateGPU対candidateCPUと両A/Aも必要。5gate testsと独立auditorで全16384raw順序/count/hash/counter、1024warm全状態、gateを照合した。

| 条件 | CPU対D236 / off | GPU対D236 / off | GPU対候補CPU | CPU AA絶対中央値 (安定round) | GPU AA絶対中央値 (安定round) | CPU/GPU同/joint |
|---|---:|---:|---:|---:|---:|---|
| c0 | +18.119% / +16.657% | +16.312% / +16.247% | -21.745% | 3.488% (3/5) | 2.741% (4/5) | 0/1/0 |
| c1 | +19.800% / +21.878% | +15.521% / +17.414% | -16.822% | 2.549% (3/5) | 3.652% (4/5) | 0/1/0 |

正は短縮。全負値/全round/全不合格と対照を保存し、call p50比への置換や有利なepoch選択、再計時はしない。fixtureでのPF gateと実走の精度は別の証拠で、実RTKの速度やfullroute/multiseed精度を主張しない。製品/既定は維持する。

## 保全

36製品source・5installed PE・PF build PE・118過去reports・58archives/12121filesを保全する。開始時は直前full-byte検証を再利用し、全manifest/report/source/PEと最新211filesを照合、計測終了後の最終時に過去archive bytesを1回再確認する。新たなnative/CUDA build・sanitizerはない。

[結果](../benchmarks/rbpf_move_plan_20261018.json)、[完了照合](../benchmarks/rbpf_move_plan_completion_check_20261018.json)、[checked caller](../benchmarks/rbpf_move_plan_checked.py)、[元の不採用guard](../benchmarks/rbpf_move_plan.py)、[実走wrapper](../benchmarks/bench_rbpf_move_plan_field.py)、[性能screen](../benchmarks/bench_rbpf_move_plan.py)。
