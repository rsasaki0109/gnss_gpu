# RBPFの親クラスタ照合を任意精度整数のままC++化 (D234, 2026-10-16)

**高多様性条件（128粒子/128初期仮説/128行/8group）のCPU paired短縮は59.14%、5/5round改善、CPU主screenはFAIL。** CPU主screenは0/6、GPU主screenは0/6だった。前回D233のC++キー生成を含むCPU/GPUを同時測定の基準にした。製品・凍結モデル・既定動作は維持し、全route/multi-seed精度や実FGO全体の速度は未承認。

## 変更と精度の維持

元の`_basin_parent_score(child,parent)`が作る共通group/satellite集合を、GILを保持するCPython C++拡張で照合する。基準は両方に存在する衛星の最小値である。子・親それぞれで`z[ref]-z[sat]`をPython任意精度整数として減算し、完全一致の個数と比較数を返す。整数を固定幅/floatへ変換せず、2**16000のgaugeや2**4096の差も保つ。builtin整数には副作用がないため、集合をsortする代わりに共通の最小衛星を探し、残りはdict順で数える。並び順によらず返す2整数は同じで、浮動小数点の加算順序は変更しない。

全入力のexact dict/tuple/intを算術前に確認する。groupはbuiltin整数2個のtuple、satellite/valueもbuiltin整数に限る。unsupported mapping/subclass/NumPy整数/bool/floatは元のPython関数を1回実行し、callback/例外を保つ。各callで元literal codeと有効な`set/sorted/len`を照合する。countの合計がPy_ssize_tに収まることもpreflightし、整数値自体にはbit長制限を加えない。

`NativeParentScoreStep`はD233のper-call viewへ静的score関数だけを差し込む。`_cluster_state`のcode objectは変更しない。support/exact keyの早期判定、親候補全件の走査、min/frac閾値、元のPython除算、ランキング、top同点時None、返却stateのidentityを保つ。子/親/keyのcacheは追加せず、更新は次callへ反映する。実runtime/moduleを上書きしない。

候補・粒子・prior・並び・RNG・FP64・2IRLS・conditional mean・FIX/hold・graphを変更しない。strict NumPy policyは元step全体へ戻し、例外までの部分state/RNGを保つ。D226の数値/転送DLLとD233のキー生成拡張はbyte単位で同じ。D231 warp/D232 DD展開との組合せは測らず、採用しない。

## 完全PF-stepの測定

6条件×5round×3group×5mode=450timed calls、warm snapshot30。D233 CPU/新CPU/新GPU/D233 GPU/D233 CPU A/Aを直列回転した。型・binding guard、全親照合、キー生成、move、scoreのpack/transfer/readback、conditional、mass/FIX、resamplingをtimerへ含める。restore/setup/warm/検証は時間外。外部extract/time/satellite/integer searchはfixture、graph solveは対象外。

自processの1許可CPU/1BLAS thread、通常priority、host/GPU非占有。paired group短縮率のround内中央値→5round中央値を使う。p50同士の比とは異なる。CPU主screenは旧CPU比10%以上、4/5round以上改善、A/A中央値abs5%未満。GPU主screenは旧GPUと新CPUの両方に同じ改善基準を要求する。判定は丸め前。全失敗・負値・対照を残し、再測定しない。

| particles/initial_unique/rows/groups | CPU p50 ms D233→新 | CPU同士paired短縮 | GPU同士paired短縮 | GPU対新CPU短縮 | CPU A/A短縮 | CPU主screen | GPU主screen |
|:---|---:|---:|---:|---:|---:|:---|:---|
| 64/6/64/4 | 19.398 → 20.531 | -7.162% | -0.146% | -4.865% | +2.854% | FAIL | FAIL |
| 64/32/64/4 | 41.298 → 37.168 | +16.597% | -4.302% | +5.153% | +1.069% | FAIL | FAIL |
| 64/64/64/4 | 53.646 → 62.831 | -6.752% | +9.229% | +4.084% | -5.328% | FAIL | FAIL |
| 128/32/64/4 | 56.054 → 54.277 | -1.106% | -1.939% | +16.711% | -11.043% | FAIL | FAIL |
| 64/6/128/8 | 39.367 → 35.604 | +0.849% | -2.505% | +15.531% | -9.108% | FAIL | FAIL |
| 128/128/128/8 | 1477.481 → 634.852 | +59.137% | +53.233% | -6.386% | -5.507% | FAIL | FAIL |

CPU主screen0/6、旧GPU同士の副screen0/6、GPU主screen0/6。A/A失敗は4/6だった。最初の4条件はcluster_relin無効で新scoreの呼出しが0である。そこでの正負の値を照合処理による高速化として解釈しない。親照合のある2条件も個別のgateで扱い、対照失敗や小さい改善を他条件の結果で上書きしない。背景負荷やallocator等に時間の変化を断定しない。

## 正確性と実走

71tests PASS。1,200ランダムな疎な子/親pair、各々のgauge/挿入順変換と引数逆転を比較した。最小共通衛星が各mapの最小と異なる場合、空/1衛星/非共通group、巨大整数、custom型/helper/callback/例外、10,000call参照数・更新反映、module/isolated ASTの2種code形を確認した。5種min/frac設定でexact key優先、低support、同点、順序、返却state identityも比較した。

PFは3backend×3seed×2cluster構成×4epochで、D233の同backendと全保存state・位置/P・RNG・move/key metadataが一致した。全450timed state hashと30warm snapshotも一致し、新しい許容差を加えない。warning3は既知config1とstrict-policy原処理のunderflow2。CUDA memcheckは13targeted tests PASS、warning1、error/leak0。新しいCUDAコードはなく、同じ数値DLLとの統合を検証した。CPU参照寿命について別child processで100call/100callbackの入力clear GC stressも通過した。これはCPU ASanや任意Python版への保証ではない。

補助validatorの初回失敗も保存した。統計auditorはコピーしたA/Aラベルが旧名だったためKeyErrorで終了し、別v2でlabelを修正して同じ測定を監査した。最初のGC fixtureはtuple freelistの影響でcallbackが発火せず失敗した。別v2で各round前に4,096個の空score tupleを保持してfreelistを消費し、GCをarmして100callbackを確認した。製品/実験実装・測定source・unit testには修正を加えず、速度の再測定もしていない。失敗script/exit/logと修正SHAをreport/archiveに含める。

実GTSAM+Tokyo run3/seed20260710/400epochの既定構成では、CPU/GPU各200、計400PF stepの保存state JSONL全行と軌跡NPZ全配列のdtype/shape/bytesが凍結D233と一致した。ただし元のcluster_relin既定0のため、新しいparent score呼出しは0であり、この実走は既定構成の非回帰確認に限る。

変更箇所を実走で通すため、元モデルの既存option`RBPF_CLUSTER_RELIN=1`をD233/D234両方に指定する別の実走を追加した。その他のcfg全項目が対照と同じことを保存configで確認し、元のtop2/min3/fraction0.8も維持する。新旧CPU/GPU各1prefixで同backend比較を行い、各200、計400paired PF stepの全保存state/軌跡bitsが一致した。新parent native callが実際に正、fallback0であることも確認した。この設定の結果はcluster機能の有無を比較する精度評価ではない。

2構成を合わせ800paired PF stepで、位置/P/FIX/hold/粒子/重み/RNG/feedbackを含む。有限scalar bitsと全数値配列bitsは一致、Python scalar NaN payloadのcanonical化は既存codecの制約として残る。source/input/external tc/binary hashを前後で検証した。実走は精度対照に限り、凍結参照や今回の単発run時間を速度の基準に使わない。全route/multi-seed測位精度gateは未完了。

実走counter（時間内訳ではない）:

```json
{
  "r3_seed0_parent_cpu": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12504,
    "extend_calls": 2110,
    "shrink_calls": 2110,
    "requested_moves": 206522,
    "unique_moves": 4363,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "canonical_native_calls": 50268,
    "canonical_fallback_calls": 0,
    "parent_native_calls": 0,
    "parent_fallback_calls": 0
  },
  "r3_seed0_parent_gpu": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12504,
    "extend_calls": 2110,
    "shrink_calls": 2110,
    "requested_moves": 206522,
    "unique_moves": 4363,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "canonical_native_calls": 50268,
    "canonical_fallback_calls": 0,
    "parent_native_calls": 0,
    "parent_fallback_calls": 0
  },
  "r3_seed0_cluster_parent_cpu": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12508,
    "extend_calls": 2007,
    "shrink_calls": 2007,
    "requested_moves": 206341,
    "unique_moves": 4260,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "canonical_native_calls": 54915,
    "canonical_fallback_calls": 0,
    "parent_native_calls": 8382,
    "parent_fallback_calls": 0
  },
  "r3_seed0_cluster_parent_gpu": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12508,
    "extend_calls": 2007,
    "shrink_calls": 2007,
    "requested_moves": 206341,
    "unique_moves": 4260,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "canonical_native_calls": 54915,
    "canonical_fallback_calls": 0,
    "parent_native_calls": 8382,
    "parent_fallback_calls": 0
  }
}
```

## メモリ・適用範囲

C++はcall中に借用参照を使い、減算結果をRAIIで解放する。返す2整数tuple以外の入力/親/key参照、native永続buffer、追加CUDA allocationは持たない。Python engineはmodule/元関数/codeとcounterを所有する。Python/NumPy heap peak、driver/JIT/stackは測定していない。既存score workspaceはcapacity512、device/host各1,143,352bytes/context。2contextのbenchmarkは各合計2,286,704bytesである。

36製品sources、5installed extensions、PF build PE、D226 DLL、D233拡張と過去28reports/13archives（1,469files）を維持する。高負荷の合成条件は親照合32,512call/stepだが、実走の有効構成は元のtop2である。親候補の多い条件の利得を既定cluster0や通常top2、全graphへ一般化しない。全graph高速化・GPU採用・既定dispatch変更は別のgateで扱う。

再現: [build/API](../benchmarks/probes/rbpf_parent_score/README.md)、[完全step](../benchmarks/bench_rbpf_parent_score.py)、[既定実走](../benchmarks/bench_rbpf_parent_score_field.py)、[cluster有効実走](../benchmarks/bench_rbpf_parent_cluster_field.py)、[全結果](../benchmarks/rbpf_parent_score_20261016.json)、[完了照合](../benchmarks/rbpf_parent_score_completion_check_20261016.json)。
