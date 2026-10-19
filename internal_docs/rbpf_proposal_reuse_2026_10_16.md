# RBPF共通候補と提案確率の再利用は採用を見送る (D230, 2026-10-16)

**前回D229を維持する。今回の方式は主速度gateを全6条件で通らなかった。** 共通候補のstagingと同じ提案分布の正規化/指数計算を呼出内で再利用する明示`ProposalReuseStep`を試した。粒子や候補を削らずに計算回数を減らせたが、完全stepで安定した短縮を確認できなかった。製品・凍結RBPF・既定設定の変更はない。

## 試した処理

D227の元move callerのASTから、stagingとweight計算の式を取り込む。source SHAを固定し、元の重み計算7文を同じ順序で別関数へ移す。候補/score indexの初出順を保つため、共通consensus/adopt/spawn一覧は最初の粒子のbase moveの後に一度だけ作る。後続粒子は同じread-only tupleを末尾に付ける。全粒子について同じ順序の候補/priorを残す。

同じ共通suffixと同じ順序のparticle-specific `(prior, score index)` を持つ場合だけ、元式のlog normalizerとprobability配列を共有する。異なるH/tagを同じHとしてcommitしない。抽選結果はその粒子自身のrecordsから取り、各粒子の`logw += mass`と元の`rng.choice`を元順で全回実行する。選択したHは元stepが粒子ごとにcopyする。

キャッシュ対象priorは正の有限builtin floatに限定する。正規化でzero/nonfiniteになった結果は再利用せず、各粒子で元の式を評価して警告を保持する。probability配列を共有できるRNGは標準`numpy.random.Generator`のみで、custom RNGは毎回配列を作る。common入力の非標準型/非有限prior等はD229 move blockへ戻す。strict/underflow-warning policyは継承した元の完全step fallbackを使い、部分RNG/stateを保つ。通常入力の型確認や管理コストもtiming内に含める。

D228の厳密なextend/shrink再利用、D229のstaging key memo、全候補/prior/RNG/commit、FP64/元の2IRLS/数値式/閾値は維持した。module globalやruntimeのmethodを一時差し替えしない。cacheはmove block内だけで、正常/例外後に次epochへ持ち越さない。native CUDA kernel/DLLとFGO graph solverは変更していない。

## 完全stepでの測定

6条件×5round×3group×5mode、450timed calls。D229 C++、候補C++、候補GPU、D229 GPU、同一D229 C++ A/Aを順次回転した。候補比較180、A/A90。時間外のliteral元scalarと候補scalarも含む42warm stateを保存した。
graph抽出・integer-search・時刻/衛星入力のみfixtureで、staging/type guard/key作成、pack/全転送/返却、条件付き位置、mass/FIX、resamplingを時間内に含む。restore/setup/warm/検証/保存は時間外。自processの1許可CPU・1BLAS thread、通常priority、host/GPU非占有。
paired短縮率は同じgroup内の比をround内中央値→5round中央値にする。候補C++対D229 C++で10%以上・4/5round改善・同一C++ A/A中央値abs5%未満を要求。GPU対候補C++も同じ主gateを使う。全条件/roundを残し、合格するための再測定は行わない。

| 粒子 / 初期固有H / rows / groups | D229 C++ ms | 候補C++ ms | paired短縮 | 改善round | A/A短縮 | 主screen |
|:---|---:|---:|---:|---:|---:|:---|
| 64 / 6 / 64 / 4 | 26.072 | 24.286 | +6.10% | 3/5 | -1.50% | FAIL |
| 64 / 32 / 64 / 4 | 24.289 | 25.013 | -7.05% | 1/5 | -8.18% | FAIL |
| 64 / 64 / 64 / 4 | 43.742 | 41.928 | -0.69% | 2/5 | -3.06% | FAIL |
| 128 / 32 / 64 / 4 | 37.178 | 37.110 | +0.81% | 3/5 | -1.11% | FAIL |
| 64 / 6 / 128 / 8 | 31.437 | 30.991 | -6.59% | 2/5 | -2.78% | FAIL |
| 128 / 128 / 128 / 8 | 993.668 | 999.195 | +2.56% | 3/5 | +1.17% | FAIL |

C++対D229、GPU対候補C++とも0/6合格。最良のC++短縮は6.10%・3/5roundで基準未達。1条件のA/A中央値は-8.18%で不合格だった。個別A/Aは全30round中19roundでabs5%を超える。p50単体の比や有利なroundから主判定を書き換えず、遅延の原因を管理コストだけに帰属しない。

同じGPU backend同士の参考比較は、64粒子/初期6H/64rowsと128粒子/初期32H/64rowsで17.33%/17.70%、各4/5・5/5round改善だった。これは事前の主gateとは別の参考比であり、最適化したC++へのGPU優位や新kernelの利得ではない。全値はJSONに残すが、この2つを根拠に採用しない。

最初の合成条件では正規化/指数計算64要求→実計算6回、staging canonical要求468→216。共通records4件を各粒子で再生成する処理を除いた。全候補数/prior/unique score数/extend/shrink実callとRNG stateは対照と同じ。計算回数の削減を全体速度倍率に換算しない。全450timed state hashと42warm stateは同じbackendのD229に一致し、候補scalarもliteral元scalarと全保存stateがbit一致した。

## 実データの確認

主速度gate不合格後は、準備した7job案を実行せず、run3/seed20260710/400epochの既存D229 C++と候補C++の2jobに絞った。これは新たな速度再試行ではなく精度だけの確認であり、元7job準備source・変更理由・実行前planを保存した。
実GTSAM/cssrlibの200PF step（有効197）で、位置・covariance・粒子・重み・FIX/hold・RNG/feedbackを含む全保存stateと軌跡NPZの全配列bytesが一致した。配列dtype/shape/bytesと有限scalarのbitsを保存する。従来codecによるPython scalar NaN payload canonical化の範囲は除く。全route/multi-seed gateは未完了。

全206,522要求moveを保ち、proposal 12,608要求→実計算850回（hit 11,758）。canonical staging要求は206,522→45,179。実走の共通suffixは全197有効stepで13recordsだった。合成の4recordsとは異なる負荷だが、実走は各1回の直列unpinned診断である。compile/cache/実行順/負荷を統制しておらず、PF call秒やrunner wall差を速度gateにしない。過去runとの時間比較もしない。fallbackは0で、extend/shrinkはD229と同じ各2,110callだった。

実走の最大cache entryは17、probability配列payloadは2,232bytes。合成全多様性の128粒子では128entry/8,160bytes。これは配列payloadだけで、Python key/tuple/辞書/H等を含むheap peak実測ではない。全てのHが異なれば確率再利用hitは0で管理費用が残る。新たなnative/CUDA workspaceは追加していない。

## 検証と記録

新76tests PASS、設定warning1。3seed×4epoch、scalar/C++/GPU、cluster/uniform/Huber/IRLS、shadow/spawn/resampling、巨大整数gauge/辞書順、maintenance、feedback commit/release、zero/negative/overflow priorの警告・部分state、strict fallback、custom RNGによる確率配列変更、抽選例外後のcache解放を検査した。
CUDA memcheckは12targeted tests PASS、設定warning1、0 error/0 leak。過去のracecheck/synccheckを今回の新規PASSに足さない。36製品source、5installed extensions、PF build PE、同じD226 DLL、過去20report/8archiveは不変。

実験用source/helper/testは再現のため保持するが、主速度gate不合格のため採用しない。D229の効果を今回の追加利得として数えず、元NumPy/nativeの既存微小差に新たな許容差を加えない。記録: [全結果](../benchmarks/rbpf_proposal_reuse_20261016.json)、[完了照合](../benchmarks/rbpf_proposal_reuse_completion_check_20261016.json)、[合成helper](../benchmarks/bench_rbpf_proposal_reuse.py)、[実走wrapper](../benchmarks/bench_rbpf_proposal_field.py)（今回の確認は`--reuse-proposals`を明示）。
