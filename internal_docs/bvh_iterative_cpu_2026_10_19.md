# D295: CPU BVH構築の再帰stack overflow修正 (2026-10-19)

**重複三角形の深い木でCPU構築が落ちる問題を修正し、製品sourceとローカルBVHバイナリへ反映した。** 分割式・sort・tie・node番号・三角形の並び・GPU計算は維持する。高速化の数値は主張しない。直前D294の反射tile配置は不採用のまま。

## 原因と修正

D294検証中、元installed PEだけでも8192個の零三角形で0xC00000FDを再現した。SAHが同点の最初のsplitを選び続けるため木が深くなり、再帰が通常stackを使い切る。分割方法を変えず、`Work(start,end,parent,right)`のvectorへ未処理範囲を積む。右を先に積んで左を先に処理することで元と同じnode番号を保つ。bounds・SAHの浮動小数点式・std::sort・比較・leaf条件は元sourceからそのままコピーした。CPU候補は通常のmain thread stackで実行し、stackサイズを増やしていない。

製品変更は`src/raytrace/bvh.cu`のCPU schedulerと、bindingsの`_bvh_iterative_build=True`という診断markerだけ。GPU境界以降は全sourceが完全一致し、tileのAPI/cache/kernelは導入しない。元sourceへの可逆投影も照合した。

## 検証

**750unique testsが通過。導入後は124testsが全成功。** 内訳は初回の既存/新規735成功、独立processで追加14成功、導入後の製品source一致1件。BVH構築、resident lifetime、LOS/multipath/chunk/cache、公開wrapper、raytrace、都市signal/reflection/diffraction等を確認した。元CPU参照だけを64MiB threadで動かし、候補は通常main threadでrandom/zero各1・64・4096・4097・8192、zero/plane/line重複各65・4096・8192について全node/permutationのdtype/shape/bitsを比較した。既存legacy90ケースも元installed PEと全返却bitsが一致し、入力不変だった。

最初のまとめ実行は735成功・14fixture setup errorであり、成功suiteへ読み替えていない。原因はcanonical名でロード済みの同じDLLを、追加fixtureが別名でもう一度初期化したこと。元log/exit/test/helper/PEを保持し、未実行だった14件だけを各DLL一度の新process2本に分けて実行した。735件と90ケースの取り直し、数値codeやtest内容の変更はない。導入後124件は、実installed PEと製品sourceを確認する別の必要な検証で、元参照の再ロードを行っていない。

GPU機械語の比較では実installed元PEが10関数、現在の元sourceを再buildすると12関数だった。**共通10関数の全SASS/encodingが一致**し、追加2関数はD295以前から製品sourceに存在するLOS frontierの2templateである。修正版12関数はD294で同じ元sourceからbuildした12関数ともすべて一致した。元の「全kernel集合一致=False」というrecordはそのまま残した。D295のCPU修正によるGPU数値式変更はなく、installed PEの由来の差も隠さない。

CUDA12.8/MSVC Release/C++20/CUDA20、default FMA、fast mathなし。D295の新しい速度測定・RTK走行・CUDA checker実行は0。D294で3checkerが成功した事実は別評価として保持する。全routeの測位精度、PF/FGO速度、GPU使用率について新しい保証にはしない。

## 再現と保全

historical projectorは製品の新CPU sourceへ適用せず、raw hashを固定した`bvh_baseline_20261019`を入力にする。新しい`generate_frozen.py`から不変の元projectorを呼べる。pytestは元のfatal fixture1ファイルだけを収集対象から外し、同じGPU関数をimportしたchecked suiteを使う。historical projection16件は同じ元source bytesへ向け、他testsの収集は変更しない。6policy testsで範囲とhashを確認した。製品sourceを改変するために、過去実行済みtestやhelperを書き直してはいない。

元installed BVH PEとconftest、元2製品sourceを変更前snapshotへ保存し、検証済PEをatomic replaceした。152既報・75archives・20100filesを保持し、元36製品sourceのうち上記2件以外、既存5installed PEとPF build PEは不変。D294を含む旧FAILの再測定・採用・判定変更はない。

[結果](../benchmarks/bvh_iterative_cpu_20261019.json)、[完了照合](../benchmarks/bvh_iterative_cpu_completion_check_20261019.json)、[CPU projection](../benchmarks/probes/bvh_iterative_build/generate.py)、[CPU exact tests](../tests/test_bvh_iterative_cpu.py)、[重複形状回帰](../tests/test_bvh_cpu_duplicates.py)、[収集policy](../tests/bvh_frozen_projection_policy.py)、[直前の不採用評価](bvh_tiled_geometry_2026_10_19.md)。
