# PF-only RTK scale-up plan — 2026-07-18

## 1. Mission

FGOを実行時に使わず、PPC Datasetの都市環境で、安全性を維持したまま
PF-only RTKのsub-50 cmカバレッジを大幅に伸ばす。

最終目標はPPC Tokyo run1/2/3の`<50cm_full%`でinuex35の
`56.7 / 69.9 / 67.9%`を超えること。中間目標では、FIXカバレッジだけで
なく、誤FIX、再捕捉時間、出力連続性、計算時間を同時に評価する。

実行時に許可するもの:

- Rao-Blackwellized PFと、粒子・ambiguity basinごとの条件付きKF;
- IMU preintegration、Doppler、TDCP、DD pseudorange、DD carrier;
- LAMBDAによる整数候補生成;
- オンラインfixed-lag particle smoothing / backward simulation;
- 3D map、ray tracing、NLOS priorを用いる独立な実験arm。

実行時に禁止するもの:

- sliding-window LM、GTSAM、その他のFGO;
- truth、hybrid RTK解、inuex35出力をproduction armへ入力すること;
- run名やepoch番号に依存する手調整;
- 同じ観測を独立証拠として二重計上すること。

## 2. Starting point and measured gap

WP23bのtrusted armはTokyo全runで24/24 FIXが正しく、false fixは0だった。
一方、`<50cm_full%`は`1.652 / 1.650 / 2.065%`で、すべてのFIXが最初の
1200 epochs内に集中した。安全性は通ったが、長時間の追尾・復旧・再FIXが
できていない。

DDPR-centered respawnでは、top-64候補の26/40 trigger epochsにoracle
sub-50 cm候補が存在した。しかし安全なDDPR設定では正しいposition cluster
massが約0.505に留まり、DDPRを0.5 mへ強めると0.626 mのfalse fixを2回
生成した。したがって次のボトルネックは候補供給だけではなく、coherent
multipath下で正しいbasinを選ぶ独立な時間証拠である。

## 3. Target metrics

### Safety invariants

- declared FIX false rate `<= 1%`を全gateで維持する;
- false fix countが少数の場合もWilson 95%上限を併記する;
- FIX判定後にtruthを参照し、判定前には絶対に参照しない;
- default/trusted armに対する意図しないfix-state差分を0にする;
- NaN、非SPD covariance、beta未消費、lineage破損を0にする。

### Coverage milestones

| Milestone | Tokyo full-run `<50cm_full%` | Purpose |
| --- | ---: | --- |
| M0 | current `1.7–2.1%` | WP23b reference |
| M1 | each run `>=5%` | repeated re-acquisition exists |
| M2 | each run `>=15%` | temporal ambiguity tracking works |
| M3 | each run `>=35%` | urban integrity/recovery works |
| M4 | `>56.7/69.9/67.9%` | beat inuex35 |

各milestoneはfalse fix safety、全epoch分母、同一設定、truth-free production
pathを満たした場合だけ通過とする。平均値だけでrun単位の失敗を隠さない。

### Operational metrics

- time-to-first-fix、outage後time-to-refixのmedian/p90;
- FIX survival length、release latency、incorrect-hold count;
- ambiguity candidate oracle coverageとMAP selection accuracy;
- gamma reliability diagram、Brier score、ECE;
- 1 m / 2 m coverage、AllRMS、95th percentile、maximum gap without output;
- wall time、epochs/s、peak memory、basins/satellite、LAMBDA calls/epoch。

## 4. Estimator architecture

Estimatorを次の4層に分離する。

1. **Observation integrity layer**
   衛星・周波数ごとにLOS/NLOS、slip、outage、generation、innovation historyを
   管理し、使用可能な観測とその分散を出力する。
2. **Temporal motion layer**
   IMU、Doppler、TDCPから、epoch間変位と共分散を生成する。絶対位置の
   DDPRとは別のevidence ledgerを持つ。
3. **Ambiguity-basin RBPF layer**
   integer assignment、lineage、conditional navigation KF、累積周辺尤度、
   integrity modeを保持する。候補のbirth/adopt/keep/release/respawnを扱う。
4. **Commit and output layer**
   `SEARCH -> CANDIDATE -> COMMITTED -> HOLDOVER -> RELEASED`状態機械で、
   calibrated posterior、独立証拠、連続時間、DD数からFIXを決定する。

各観測更新は`source_id`、対象epoch、beta、log evidence、使用衛星集合を記録し、
二重計上をaudit可能にする。position cluster massはsingle-link clusteringを使わず、
non-chaining ballまたは明示的mixture componentで集約する。

## 5. Work packages

### WP24 — evaluation contract and evidence ledger

最初に評価の地盤を固定する。

- WP23b runnerから共通epoch trace schemaを分離する;
- observation source別log evidence、generation、candidate rank、cluster identity、
  commit state、guard distancesを保存する;
- deterministic replayを作り、同じtraceからcommit policyを再評価可能にする;
- full denominator scorerとFIX calibration reportを1コマンド化する;
- Tokyo run2/1200をsmoke専用に降格し、選択と検証を分ける。

Gate:

- trusted WP23bの24/24 correct、0 false、full scoresを再現;
- replayとonline判定がbit-identical;
- observation sourceのbeta総和と二重計上auditが全epochでpass;
- CPU/GPU、repeat run間のfix-state mismatchが0。

### WP25 — temporal ambiguity RBPF

単一epoch gammaではなく、持続する整数状態のsequence posteriorを扱う。

- ambiguity assignmentにstay/slip/outage/re-entry transition probabilityを導入;
- basinごとに複数epochのwhitened TDCP/DDCP innovation historyを保持;
- fixed-lag `2/5/10 s`でancestor samplingまたはFFBSiを行う;
- position clusterではなく、assignment lineageとmotion consistencyでmassを統合;
- commitにはminimum dwell、posterior odds、independent temporal residualを要求;
- hard releaseとsoft holdoverを分け、誤った整数を長時間保持しない。

Gate:

- synthetic slip/outage/multipath scenarioでcorrect lineageのsurvival `>=95%`;
- WP23b respawn ablationの正解候補を選ぶ率がsingle-epoch MAPを上回る;
- Tokyo full runのtime-to-refixが有限になり、各runでM1を達成;
- false fix `<=1%`、sigma-DDPR 0.5 mのようなconfidence製造は禁止。

### WP26 — independent relative-motion evidence

DDPRと同じ誤差を再利用せず、時間方向からbasinを判別する。

- GPS L1から開始するslip-aware TDCP displacementとcovariance;
- Doppler velocity、IMU preintegration、TDCPのinnovation cross-check;
- per-basin navigation KFへposition/velocity/heading-error/biasを条件付け;
- static/turn/acceleration regimeごとのprocess-noise adaptation;
- TDCPを絶対anchorとして使わず、候補間のrelative consistencyにだけ使う;
- correlationを測り、DDCPとTDCPの重複成分をwhitenまたはjoint updateする。

Gate:

- clean intervalsでTDCP displacement NISが校正範囲に入る;
- blocked intervalsでwrong basinとcorrect-oracle basinのtemporal log Bayes factorが
  有意に分離する;
- IMU/Doppler/TDCPのleave-one-source-out ablationでgainの由来を説明できる;
- WP25に統合してM2を目指し、安全性を維持する。

### WP27 — satellite integrity and robust carrier model

coherent multipathを単なるGaussian noiseとして扱わない。

- satelliteごとのlatent mode `{clean, biased, blocked, recovering}`;
- C/N0、elevation、lock time、geometry-free、Melbourne-Wubbena、innovation
  persistenceからmode transitionを推定;
- DD pivot汚染を検出し、multi-pivot consensusまたはpivot mixtureを導入;
- common-mode/coherent shiftを低ランクbias stateとして条件付け;
- partial ambiguity fixingを衛星品質と情報量で選ぶ;
- 3D BVH/PLATEAU priorは独立armで評価し、coverageのある場所だけfusionする。

Gate:

- false-fix直前の衛星集合をtruthなしでvetoまたはdownweightできる;
- clean observationを過剰除外せず、usable DD countの低下を報告する;
- run-local tuningなしでTokyo/Nagoyaに同一policyを適用;
- WP26までのarmに対してM2以上、可能ならM3へ進む。

### WP28 — outage recovery and active hypothesis management

長いurban outage後に再び正しいbasinを作り直す。

- outage長とgeneration changeに応じたSEARCH/RECOVERING state;
- DDPR-centered seed、motion-propagated seed、multi-pivot seedを別proposalとして保持;
- proposal sourceごとのprior massをcalibrateし、top-K rankだけで候補を切らない;
- diversity-aware basin cap、assignment/position両空間のdeduplication;
- correct candidateが供給されたのに消えるepochをsurvival auditする;
- recovery中はFIXを急がず、候補探索と出力品質を分離する。

Gate:

- artificial and natural outageでcandidate recall `>=90%`を先に達成;
- correct candidate conditional survival p90 `>=5 s`;
- time-to-refix medianをWP23bより短縮し、incorrect holdoverを0にする;
- full-run M3を目標とする。

### WP29 — GPU scale-up and real-time budget

精度が確認されたアルゴリズムだけを最適化する。

- multi-seed/top-K LAMBDAを真のbatchとしてGPU化;
- basin conditional updateをSoA化し、batched small-matrix solveへ移す;
- shared RINEX/geometry/atmosphere cacheを維持;
- adaptive basin budgetとproposal scheduling;
- deterministic fast modeとaudit modeを用意する。

Gate:

- reference armとのfix-state mismatch 0、position delta toleranceを明示;
- 5 Hz inputに対して平均`>=5 epochs/s`、p99 latencyを計測;
- peak memoryが長時間runで増加しない;
- full PPC six-runを一晩以内に再現できる。

### WP30 — locked benchmark and release candidate

- 設定を凍結してPPC Tokyo/Nagoya全6runを評価;
- UrbanNav deep-urbanへ同じcoreを移植し、mean horizontal `<=3.5 m`を評価;
- safety case、failure taxonomy、calibration、runtimeを公開可能な形に整理;
- inuex35、hybrid floor、WP23b、各leave-one-component-out armと比較;
- reproducible command、environment manifest、artifact checksumsを保存する。

Gate:

- M4または、M3以上かつ残差のroot causeが測定されたhonest negative;
- PPC全6runでfalse fix `<=1%`;
- truth-free、FGO-free、同一設定、full denominatorをauditで証明;
- report、tests、compact evidence artifactsをcommitする。

## 6. Dataset and anti-overfit policy

既にTokyo全3runを観測済みなので、厳密な未見holdoutとは呼ばない。今後は:

- **smoke**: Tokyo run2 first 1200（機能・回帰のみ）;
- **development**: Tokyo run2/run3の事前固定区間;
- **safety validation**: Tokyo run1 full;
- **transfer validation**: Nagoya run1/2/3 full;
- **final report**: PPC全6run、設定凍結後に1回実行。

区間境界、metric、threshold sweep範囲を実験前にspecへ書く。validation結果を見て
thresholdを変更した場合、その結果はdevelopmentへ降格し、新しいvalidation gateを
設ける。run別定数やepoch blacklistはproduction policyに入れない。

## 7. Execution rules for every WP

1. `internal_docs/task_wpNN_*.md`に仮説、変更範囲、gate、repro commandを書く。
2. synthetic/unit testを先に作り、次にshort real-data smokeを行う。
3. candidate supply、posterior selection、commit policyを別々に計測する。
4. 失敗した設定はdefault-offにし、negative resultとroot causeを残す。
5. safety gate通過前にfull six-runへ計算資源を使わない。
6. 各WPを独立commitし、push/PRはユーザーの明示許可まで行わない。

## 8. Immediate next sprint

次はWP24から開始する。最初の5 deliverablesは:

1. common epoch trace dataclass/schema;
2. evidence ledgerとduplicate-source audit;
3. online/replay commit-policy equivalence test;
4. trusted WP23b full-result reproduction manifest;
5. WP25用のwrong-vs-correct lineage temporal diagnostic。

このsprintでは新しいFIX率を主目標にしない。後続の時間モデルを安全に比較できる
評価基盤を完成させることをpass条件とする。
