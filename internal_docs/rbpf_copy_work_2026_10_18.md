# 実走で仮説辞書の複製と所有権を調べる (D278, 2026-10-18)

Tokyo run3/400RTK epochの元D236 GPU/c0で、仮説copy17298回・内側dict200899個・整数entry888818件を観測した。一時候補の変更しないgroupを共有して省ける内側dictは30307個、全体の15.086%だった。事前に定めた50%の処理量triageに届かず、共有化や速度screenへは進まない。これは開発優先度の判断であり、速度向上率や速度上限ではない。

新しい読取り専用observerを追加し、21tests PASS、全200PF保存状態・RNG・feedback・従来metadata・全測位軌跡/cfgがgoldへ一致した。数値関数・粒子/候補数・FP64・IRLS・prior・閾値・RNG・graph・製品/既定を変更していない。

## 処理量の内訳

| 呼出し箇所 | copy回数 | 内側dict | 整数entry | 一時候補で変更しない内側dict |
|---|---:|---:|---:|---:|
| extend一時候補 | 852 | 9775 | 41320 | 7783 |
| shrink一時候補 | 2110 | 24634 | 111332 | 22524 |
| 選択Hの粒子確定 | 12608 | 147033 | 653206 | 0 |
| resample | 1728 | 19457 | 82960 | 0 |
| 合計 | 17298 | 200899 | 888818 | 30307 |

削減候補のentryは134336/888818（15.114%）。これは`_copy_H`が複製する内側dict/整数entryの数であり、全Python allocation数、heap bytes、staging時間や全PF時間の比率ではない。外側dictも全て除去できるという意味ではない。D277の小さなfixtureではなく、元の実route処理を呼出し箇所別に数えた。

`_copy_H`のexact code objectだけを監視し、call frameのHから元のdict順・任意精度整数・空group除外を保持したsignatureを作る。returnで元と同じ結果と外/内dictの独立性を検査する。計17298copyで不一致・source alias・unsupportedは0。元のcopy自体は全て実行する。source/code/module/runtime methodをpatchせず、epochを越すH/key/matrix cacheや辞書共有を追加しない。

temporary削減量は監視対象の元`_extend`/`_shrink_worst` code identityを登録して集計する。extendではidxCのtarget group、shrinkでは削除対象groupを変更ありと数え、それ以外の非空groupだけを候補とする。同名の別functionは分類しない。entry/returnの観測やsignature検査はoverheadを含むため、既存runnerの秒数を速度比や上限へ換算しない。

## 所有権と全writerの確認

凍結source SHAを固定し、copy呼出し6箇所（990/1163/1567/1683/1726/1924）、H属性store3箇所、zmapへの全subscript store/mutator callをASTで照合した。周辺のaliasとwriterは以下のように扱う。

- `_maintenance`は粒子の内側dictから衛星を直接popする。epochを越す共有は、別粒子や保持したcluster priorの変更へつながる。
- extend/shrinkは全groupをcopyしてから一部groupを書き換える。読取り専用stagingの間に限った共有には余地があるが、確定までの所有権管理と独自guardが必要になる。
- `_adopt`/`_adopt_map`、debug truth、部分合意、leave-one-outは新規のローカル辞書を作る。
- 選択後の粒子確定copyは、同じstaged Hを選んだ粒子同士を分離する。resampleも同じancestorを複数回選べるため分離が必要である。古いParticleを外部が保持する可能性もあり、無条件の所有権移譲はしない。
- shadow/clusterのHは独立copyで保持する。shadowのscore/age更新やcluster containerのclearはH内容の書換えと区別する。feedbackのgraph callbackを越す共有安全性は推定しない。

全200callの入口/出口でparticle・cluster・shadow Hを列挙し、観測したouter/inner dictの重複0を確認した。これはこの境界の観測であり、任意外部保持参照・custom mapping・同時変更に対する全プログラムalias証明ではない。今回は共有を実装せず、元の分離境界を維持する。

## 精度と試験

21testsはcounter/order/空group/巨大整数/custom mapping・alias検出・同名caller誤分類防止・元例外/既存profilerの保持とcleanupを検査する。scalar/nativeCPU/GPU×2seed×2cluster×4連続epochで全状態・RNG・feedback・従来metadataのbitsと数値function code identity、GC policyを照合した。

実走はTokyo run3/seed20260710/GPU/c0の新しい観測1本。400RTK epoch/200PF call（197score/3skip）を、元PF-return境界のgoldへJSONL全行literal比較し、trajectoryの全array dtype/shape/bytes、cfg、全従来metadata、入力/外部source/PEを照合した。root keyやrtk状態を取り除かず、許容差も広げない。scalar NaN payload codecの既存制約は継承する。追加GPU転送0、scalar fallback0、既存profile callbackがあれば置換せず拒否し、通常/例外終了後は所有したcallbackを解除する。GCの有効状態・閾値を変えない。

この実走は読取り専用の処理量/精度観測で、過去に失敗した候補の再計時ではない。性能採用は0件。速度に有効な新候補を裏づけるまで、追加の共有化や微調整を積み上げない。

## 保全

36製品source・5installed PE・PF build PE・116過去reports・57archives/11910filesを保全する。開始時は直前の独立full-byte検証を再利用し全manifest/report/source/PEと最新234filesを確認、最終時に過去archiveのbytesを1回再確認する。新たなCUDA/native build・sanitizer・製品/既定変更はない。

[結果](../benchmarks/rbpf_copy_work_20261018.json)、[完了照合](../benchmarks/rbpf_copy_work_completion_check_20261018.json)、[観測器](../benchmarks/rbpf_copy_work.py)、[実走wrapper](../benchmarks/bench_rbpf_copy_work_field.py)。
