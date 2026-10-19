# 条件付きextendの同一行列LU再利用 (D274, 2026-10-18)

採用を見送る。完全PF2880timed/192warmでCPU主 0/8、GPU同backend 0/8、GPU joint 0/8。標準条件0は両backendで基準未達。52機能testsと全3072記録の同backend状態・RNG・feedback・従来metadataはbits一致した。fixture内の一致を実データ測位精度や全RTK速度の確認とは扱わない。

## 変更

D236の元extend内の1か所のsolve呼出しだけを置換し、元へ戻したAST全体の一致を確認する。元のsupport抽出、QHH/QCH gather、右辺差分、条件付き平均の行列積/差分、丸め、任意精度整数の差と辞書copyを維持する。D264のsupport/assembly変更、D273の候補登録一括化を併用しない。

同じ行列寸法・全binary64 bytesのときだけ、最初のLU/pivotをcall内で共有する。仮説ごとの右辺は毎回独立に元ライブラリのDGETRSへ渡す。逆行列・複数右辺一括化・値の近似・supportだけによる同一視はない。signed zeroや1ULP、次callで変わる行列を区別する。全候補/粒子/観測/prior、元の尤度/IRLS、重み/choice/RNG/FIX/feedback/graphは同じ。

**測定fixtureでは再利用hitが0だった。** 全candidate1024call（timed960/warm64）のsolve33792回は全て初回分解で、元のmove reuse後には同一行列が残らなかった。つまり完全PF screenは分解方式とkey管理の追加費用を測っており、再利用の実効的な利得を示す試験にはなっていない。別の機能testsでは同一行列への異なる右辺と実pivot交換の再利用pathを実行してbits一致を確認している。実走の重複率は今回未測定である。今後の同様の案は、速度suiteの前に実入力で共有対象が残るかを構造counterで確認する。

NumPy2.3.4が使う同じSHAのILP64 LAPACKを明示的に使用し、DGETRFとDGETRSを分離する。固定Windows/Python/NumPy buildの実験で、将来版・別BLASへの保証ではない。finite bounded float64の最大128次元・128行列だけを受理する。変更されたsolve関数、未対応型/shape/容量/非有限・範囲外入力、特異factorや非有限解は元solveへ戻す。親controllerのstrict NumPy policyは元whole stepへ戻り、途中state/RNG/例外を保つ。

## 精度の照合

52tests PASS。最初の40testsでは1/2/3/7/11/32/64/128次元の一般/対称行列について、異なる右辺・C/F/逆stride・返却配列書換え後の再利用を元solveへbytes照合する。入力非変更、signed zero/1ULPで別key、次epochへの非保持、singular/NaN/Inf/巨大/微小/subnormal/overflow解、custom solve、容量/型/shape fallback、cleanupも確認した。CPU/GPU/scalar×2seed×2cluster×4epochでは全PF状態/位置/P/RNG/feedback/従来metadataが元D236とbits一致する。別fileの12追加testsでは実際のpivot交換7寸法と悪条件Hilbert4寸法、factor再利用後の元RNG例外時の途中state/乱数一致とcache解放を確認した。元の許容差を広げていない。

実装/試験は初回数値実行後固定。benchmark準備初版は、削除した空文字列の逆置換が曖昧なためsource復元assertで停止した。別pathの準備scriptで各編集のoffsetを保持し、元benchmark全文への復元を確認した。失敗は数値実行前で、元preparerを保存する。測定済み実装・試験や完了した速度測定は変更/再試行していない。

## 完全PFの速度

8条件、元/候補CPU/GPUとCPU/GPU A/Aの6mode、4初期RNG seed×5round×3group。全2880timed/192warmを保存する。4stepのinteger-ns合計/group→3group中央値→5round中央値の短縮10%以上・4/5改善round・対応A/A abs中央値5%未満を要求する。GPU jointは候補CPU比と両A/Aも必要。正は短縮。factor hit列は4warm seedの合計で、速度や実データの削減率へ換算しない。

| particles/unique/rows/groups,cluster,feedback | CPU対元 | GPU対元 | CPU A/A | GPU A/A | factor hit/solve | CPU/GPU同backend/joint |
|---|---:|---:|---:|---:|---:|---|
| 0: 64/6/64/4,c0,fb1 | -7.450% | +1.439% | +14.058% | +1.942% | 0/16 | False/False/False |
| 1: 64/32/64/4,c0,fb1 | -15.167% | -17.734% | -0.741% | -7.162% | 0/120 | False/False/False |
| 2: 64/64/64/4,c0,fb1 | -36.958% | +1.721% | -28.229% | +12.320% | 0/248 | False/False/False |
| 3: 128/32/64/4,c0,fb1 | +7.617% | -21.598% | -4.954% | +0.128% | 0/120 | False/False/False |
| 4: 64/6/128/8,c1,fb1 | +1.805% | -8.472% | +0.169% | +1.296% | 0/16 | False/False/False |
| 5: 128/128/128/8,c1,fb1 | -4.755% | -6.103% | -9.739% | +1.725% | 0/504 | False/False/False |
| 6: 64/6/64/4,c0,fb0 | -12.902% | -1.471% | -4.711% | -1.337% | 0/16 | False/False/False |
| 7: 64/6/64/4,c1,fb1 | -2.020% | -3.289% | -7.819% | +3.073% | 0/16 | False/False/False |

型/有限guard・全行列bytes key・LU保持/解放・DGETRF/DGETRS・元の全PF tail/score転送と完了待ちを含む。restore/deepcopy/RNG生成/callback再bind/hash/照合/gzipは外。許可CPU1本/BLAS1/normal priority/非占有、CPU/GPUは直列でrotate/reverseする。全3072rawの順序/件数/hash/counterと192warm全状態を独立auditした。観測抽出/整数探索/時刻/衛星/graph solveはfixture境界である。A/A不成立と遅化も保存し、原因を推測で断定しない。

追加fieldへ進む事前条件はCPU主またはGPU同backendが2/8以上、かつそのbackendの標準条件0も合格すること。標準条件が未達のため追加field0状態。LU回数の削減を全体性能・GPU化率の改善として扱わない。

## メモリと保全

保持payloadは各entryで全行列key bytes＋同寸法float64 LU＋int64 pivot、最大128entry/128次元。成功/例外/skip/strict fallback後にcacheを解放し、次epochへ数値を持ち越さない。Python key/辞書/ctypes object/allocatorを含むheap peakは未測定。新しいCUDA数値source/転送/buffer/sanitizer実行は0、元scorerのhost/device各1143352bytesは維持する。

36製品source・5installed PE・PF build PE・108既報・53archives/10866filesを保全する。製品/既定変更はない。任意namespace変更や同時input書換え/OOM時部分状態の一般保証はない。scalar NaN payloadの既存codec制約を継承する。

[結果](../benchmarks/rbpf_extend_factor_20261018.json)、[完了照合](../benchmarks/rbpf_extend_factor_completion_check_20261018.json)、[実装](../benchmarks/rbpf_extend_factor.py)、[完全PF](../benchmarks/bench_rbpf_extend_factor.py)。
