# 登録計画の実データ連続RTK評価 (D280, 2026-10-18)

事前の複合速度gateは不通過。個別の短縮値だけで実RTK高速化を採用扱いにしない。 新しい計測・判定19tests PASS。24prefix/9600RTK epoch/4800PF callを完走し、全24測位軌跡arrayのdtype/shape/bytes、最終PF状態・RNG、cfg/入力/source/PEを照合した。計算式・観測・候補・粒子数・閾値・FP64・RNG drawを変更していない。

## 比較対象と実行

D279のchecked登録計画を変更せず使う。既存D236 GPUのreferenceと同じ実装AA、同じ新adapterで計画だけoff、candidateの4通り。offは元の登録blockをそのまま実行し、candidateは全logical候補を残して純粋な登録結果のみcall内再利用する。D279の初期失敗・33tests・200途中状態の完全一致・PFベンチマークのGPU同backend 2/2通過はそのまま保存し、同じPF速度測定を取り直していない。

Tokyo run3/seed20260710/GPU/cluster0/400epochの連続実行。最初に各armを1回ずつ計4回warm実行し、同じ新Numba cacheをその後の5round×4armで共有する。最初のcold起動も保存するがwarm4回は全て速度判定から外す。timed中にcacheのpath/bytesが準備完了時から変わればwarm-cache判定を不成立にする。cache 56files、今回のcache policy valid=True。cache loadや非cache JITの時間は実行に含め、cache不変だけからcompilation全廃とは主張しない。

最初の4roundは各armが各位置を1回通るWilliams順、5round目も事前固定。各childは同じ許可CPU1個へ固定、BLAS1、通常priority・通常GC、ホスト/GPUは非専有。別childとepochごとに交互に進めるbarrierや、GC/cycle/nested PF observerは使わない。全armに同じ軽い3時計と最新PF returnの参照保持を加える。stateの複製/metadata encode/圧縮保存はdriver完了後だけで、速度区間へ入れない。時計recordの小さいallocationによる観測負荷は残る。

## 速度の判定

正は候補の短縮率。各roundの整数nsを先に足し、reference対candidateとoff対candidateの差を求める。全400処理本体3%以上、phase2処理5%以上、driver3%以上の中央値、各4/5round以上の正の短縮を全て要求する。各境界のreference/AAは絶対差中央値5%未満かつ4/5round以上で絶対差5%未満も要求する。両対照・3境界・精度・保全・cache条件を全て通過した場合だけ合格。signed A/Aの相殺、遅いepoch除外、CPU時間への後付け切替、再実行はしない。

| 境界 | 対reference短縮中央値 | 対off短縮中央値 | A/A絶対差中央値 / 5%未満 | 個別gate |
|---|---:|---:|---:|---|
| 全400 RTK処理本体 | -12.049% | +4.282% | 18.892% / 2/5 | False |
| PF稼働phase2の200処理 | -13.599% | +3.395% | 5.919% / 1/5 | False |
| 連続driver実行 | -6.962% | -5.996% | 15.035% / 2/5 | False |

複合速度gate=False、cacheも含む最終速度判定=False。全warm/timedの各epoch wall/process CPU/thread CPU、phase1を含む全集計、5round個別値をJSONへ保存する。CPU時間や親subprocess経過時間は診断値で、wall gateの代用にしない。

既存同士のthread CPU絶対差中央値も全400で16.354%、phase2で6.515%あった。candidate対referenceのthread CPU短縮中央値は全400 -11.189%、phase2 -9.338%。CPU時計でも揺れがあり、単純なwall待機差を差し引いて速度合格へ変換しない。原因の切り分けは未完了で、scheduler/GC/GPU待ちなどの因果はこの観測だけから断定しない。これは保存済みrawの集計だけで、新しいtimingは0件。

RTK処理本体は元のprocess全体であり、観測decodeなどdriver側処理を除く。driver時計はadapted main呼出しからreturnまでの連続区間で、setup/入力hash・decode/graph/PF/ログ/trajectory書込み/teardownを含む。module importと終了後のstate/performance encodeは除く。この境界を端末起動からの時間やfullroute throughputと同一視しない。

## 精度と保存境界

24軌跡と24最終stateを独立auditで全照合。途中状態の保存は速度試験では行わず、D279の別精度実行200状態を参照する。新しい最終stateはdriver完了後、古いgoldはPF return直後であり、親RTKがhold/navを後処理できる。事前planどおり旧goldへの最終照合だけroot `rtk` を除き、残りの全PF状態・粒子・重み・RNG・出力をexact比較する。新しい全24最終stateはroot `rtk` を含めて完全一致を要求した。後付けでkeyや許容差を変えていない。

最終caller metadataのうちD279で明示した3つの実作業mapを整合検査し、それ以外は順序込みexact。offは全legacy metadataもexact。各candidateは197planned/0fallback、logical登録206522→実登録9751、offは0planned/197literal fallback。候補や数値moveの削減ではない。

36製品source・5installed PE・PF build、120過去reports・59archives/12370filesを保全する。新しい数値/CUDA kernel、転送、製品/既定変更はない。一つのroute/seed/GPU/cluster設定でのprefix検証であり、fullroute/multiseed精度、CPU対GPUの優位、GPU稼働率は新たに検証していない。

[結果](../benchmarks/rbpf_move_plan_route_20261018.json)、[完了照合](../benchmarks/rbpf_move_plan_route_completion_check_20261018.json)、[計測caller](../benchmarks/bench_rbpf_move_plan_route.py)、[schedule/gates](../benchmarks/rbpf_move_plan_route.py)、[tests](../tests/test_rbpf_move_plan_route.py)。
