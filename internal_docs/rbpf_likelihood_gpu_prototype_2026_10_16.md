# RBPF融合尤度GPU試作 (D226, 2026-10-16)

元の`RbpfFgo._loglik`を対象に、1block/固有仮説のCUDA融合kernel、同じ式のC++ CPU版、明示batch APIとsource固定adapterを実装した。**研究用の明示APIまで。既定・公開凍結runtime・製品バイナリは変更しない。**
候補生成やstepを自動変更せず、実際に要求された仮説だけを受け取る。gauge-invariantな元cache keyで重複を除き、最初の要求順にscoreを戻す。graph/particle/RNGを変更しない。

残差・unheld uniform/Gaussian・whitening・group nuisance・3x3 full prior・初期solve・元2回IRLS・mixture/Laplace尤度を1launchにまとめる。LUはpartial pivotを使い、微小な非対称も残す。新jitter/variance floor、候補削減、整数normalizerの近似は追加しない。Qbは元尤度自身が読む対角だけを使い、他のRBPF処理のfull covarianceを変更しない。

## 結果と採否

3つの独立source/planを保存した。最初は6H2D+2D2H、次は入力/出力各1転送のpinned版、最後は同じ2転送のpageable版。数値sourceとSASS全5144命令は3版で同一で、測定返却bitも全transport間で同一。
各sourceで12条件、5round×3group。最初の2実験はscalar/C++/GPU/scalar-AA、最後は3GPU transportも同じgroup内で交互比較する。計2520timed calls、1440candidate比較、540scalar-AA比較。全raw/計画/失敗を残し、合格のための再試行は行わない。
各callはdedup・DD/cluster lookup・pack・毎epoch全入力転送・GPU計算/host返却/scatterを含む。capacity確保・warm・hash・診断は時間外。1許可CPU・1BLAS thread、通常priority、host/GPU非占有。主指標はpair短縮率のround内中央値→5round中央値で、絶対p50の比とは別。

| 64観測/4group unique数 | 元NumPy ms | C++ CPU ms | 2-copy GPU ms | GPU対C++ paired短縮 | 改善round | A/A短縮 | screen |
|:---|---:|---:|---:|---:|---:|---:|:---|
| 16 | 4.890 | 0.771 | 0.819 | -7.00% | 1/5 | -3.07% | FAIL |
| 21 | 7.051 | 1.044 | 0.979 | 5.27% | 5/5 | +0.63% | FAIL |
| 32 | 9.717 | 1.360 | 1.198 | 12.09% | 5/5 | +0.38% | PASS |
| 64 | 21.661 | 2.745 | 2.344 | 17.31% | 5/5 | +1.53% | PASS |
| 128 | 43.190 | 5.736 | 4.487 | 13.14% | 3/5 | -0.20% | FAIL |
| 512 | 207.824 | 27.902 | 36.598 | -29.18% | 1/5 | +0.82% | FAIL |

**32/64 uniqueの64観測条件でGPU対C++が約12%/17%短縮。公開64粒子の代表約21 uniqueでは約5%差で、10%gateを満たさない。** 元NumPyに対してはC++一括化自体の効果が大きい。単にPython overheadを減らした効果をGPU固有の効果と混同しない。新しい自動閾値や既定GPU化は入れない。
128観測/8groupも全6サイズを保持する。最終比較ではC++ CPUを安定して10%以上上回る条件はなく、512 uniqueはscalar A/Aも不合格。128 uniqueの64観測もGPU改善3/5roundで不合格。大きければ必ずGPUが速いとは扱わない。
初回pinned単独測定は遅かったが、最後の同時比較では他のGPU版も遅い条件があり、host staging方式だけを原因と断定しない。scalar A/AはGPUの全変動を測る対照ではない。

## 数値・安全性

全timed fixtureの元NumPyとの差はLL最大9.09e-13。元NumPyとはbit完全一致ではない。3seed×rows/groups/IRLS/unheld/shadowの54診断では、各backend432固有conditionalを途中値まで照合し、conditional offset最大差4.22e-15、final normal最大差7.46e-11、inverse normal最大差1.11e-16、logdet最大差2.84e-14だった。inverse normalは尤度の最終IRLS行列の逆で、完全NavState posteriorではない。
half-cycle同点と±2^-48/±2^-40/±.1cycleの7ケースは全native backendで元のstable rankingを維持した。閾値近傍の全ケース、実FIX/trajectory/RNGの一致や新しい測位精度を証明したとは扱わない。
元数値関数をASTから無改変で取り出すoracleを使い、GTSAM importを模擬しない。共有72testsは中間sol/Lam/logdet、uniform/Gaussian、2IRLS/IRLSなし、個別prior、group順、strict/underflow-warning/exception、dtype、非対称prior、singular、所有権、容量、odd-alignmentを確認。最初の版65PASS、後続2版は各72PASS、設定warning各1。
pageable版memcheck/racecheck/synccheck各16PASS、error/hazard/leak0。最初のseparate版もmemcheck16PASS・error/leak0。checkerではCPU availabilityを引き継ぎ、benchmark pinとは分けた。前回D225の91testsや製品PF試験を今回の新規PASSとは数えない。
未測定buildのerror診断初期化修正、最初のstrict試験63PASS/1FAIL（元実装もunderflow例外となる期待値誤り）を保存した。数値閾値を緩めていない。

## メモリと呼出し

capacity512の明示device payloadは各版1,143,352B。2-copy版はさらに同量のhost stagingを保持し、pinned版はpage-locked、pageable版は通常heap。context/driver/stackやPython packのpeakはこの値に含めない。最終6mode比較だけ3workspaceを同時保持した。
kernelは56registers/thread、shared30,904B/block、stack32B/thread、spill0。128rows/128pairs/8groups/capacityはaccelerate可能範囲で、上限外は全候補を元scalarで計算する。cross-epoch数値cacheは追加しない。
新モデルを旧式で計算しないよう、adapterは凍結source SHAが違うruntimeを拒否する。実stepへ組み込む前にmoveを再生成せずbatch化するcaller設計、全state/離散/RNG比較、元の全区間multi-seed/route精度とwhole性能gateが必要。今回のcomponent screenから実測位器全体の短縮は主張しない。

記録: [全raw/診断/計画JSON](../benchmarks/rbpf_likelihood_gpu_prototype_20261016.json)、[完了検証](../benchmarks/rbpf_likelihood_gpu_prototype_completion_check_20261016.json)、[build/API/再現](../benchmarks/probes/rbpf_likelihood_batch/README.md)。
