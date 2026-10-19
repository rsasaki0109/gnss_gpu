# D257: 同期修正済みregister LUの区間計測 (2026-10-17)

前回のblock-element分担は上積み不足だったため、次の実装に先立ってLU内部を区切って測った。製品・既定・測位モデルは変更していない。最適化採用や新たな数値処理のGPU移行はない。

## 比較と較正

全394実走保存入力（cluster_relin0/1各197）で7モード、5round×3group、41370timed/2758warm。同期修正版の凍結DLL、同一DLL内の同期修正版、trace転送だけの対照、区切り同期だけを追加した対照、時刻計測版、凍結版AA、計測版AAを交互に実行した。各roundは3組の197件合計比の中央値、結果は5round中央値。全7対比の絶対値5%未満を事前基準とし、合格は1/2条件。計測完了後の再計測なし。

以下は計測オーバーヘッド（正は遅化）。各round値は結果JSONに残す。

| 対比 | c0 | c1 |
|---|---:|---:|
| total_profile_overhead | +6.351% | +2.649% |
| markers_vs_barrier_control | -1.015% | +1.165% |
| barriers_vs_copy_control | +0.916% | -0.620% |
| copy_control_overhead | +3.353% | +1.205% |
| original_control | -1.188% | -0.832% |
| gpu_AA | +1.269% | -0.785% |
| trace_AA | -0.326% | -1.420% |

c0はtotal上乗せ6.351%が基準を超えて未較正。c1は全7対比が基準内で較正通過。下記c0の割合は参考診断値とし、c1も追加同期を含む計測版の経過サイクルである。未計測版の時間配分や改善上限を断定しない。

全score時計には従来scoreの準備・dedup・packing・CPU/GPU dispatch・転送・kernel・同期・返却に加え、profile選択/query/trace readback/resetを含む。状態復元、warm、hash、trace集計・圧縮保存は時計外。BLAS1・許可CPU1個・通常優先度、非占有環境。異なる実験の絶対時間とは比較しない。

## LU区間の診断値

dimension11のみ。各block内の3回（Huberなしは1回）の区間差を合計し、LU内部totalで割った割合のblock中央値を求め、その後call間の中央値を取る。c0/c1のtimed n11 block traceは200550件、粗い全dimension block traceは246870件。各項目を独立に中央値にするため表の合計は厳密な100%を要求しない。

| 区間 | c0 | c1 |
|---|---:|---:|
| load_guard | 3.615% | 3.625% |
| factorization | 50.708% | 50.647% |
| store_logs_status | 7.202% | 7.210% |
| rhs_permutation | 1.600% | 1.600% |
| forward | 3.701% | 3.707% |
| backward | 32.964% | 33.020% |
| validation_stores | 0.218% | 0.218% |

全dimensionの粗いsolve区間比は42.450/42.546%。n11のsolve区間のうち内部markersの外側は1.085/1.080%で、warp dispatch・caller block barrier待ち等を含む。同期単独の純時間に分解したものではない。

8個の各境界にはfull-mask syncwarpを前後に追加し、区切り同期だけの対照にも残す。元のstatus読み書きを順序付ける2個のsyncwarpも保つ。clock64の差にはstall・time-slicing・待ちが入る。異なるSM間の時刻差、kernel wall time、占有率、全PFのGPU化率としては使えない。[CUDA時刻関数](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#time-function)、[同期関数](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions)。

c1の計測版では分解が約51%、後退代入が約33%で、前進代入やRHS並べ替えより優先候補になる。前回のbroadcast cache案は全scoreの上積み不足だったため、同じ局所変更を再計測して採用しない。次は共有メモリの行容量を実入力に合わせる独立案を候補とする。両条件とも181/197caseが64行以下、全197caseが72行以下で、現状は128行用scratchを常に確保する。小容量用の別kernelへ全行が収まる場合だけdispatchし、観測を省略せず独立仮説の並行実行を増やせるかを検証する。この案はまだ実装・測定しておらず、速度や実行block数は予測していない。

## 正確性と検証範囲

311tests PASS。raw7872行列（n11は2357）で元の汎用LU・追加同期対照・計測版のLU全要素、pivot、solution、logdet、statusがビット一致。3..11次元、非対称、pivot tie、途中failure、NaN/Inf、極小値、後退代入overflow、符号付きzero、相殺で演算順の違いが現れる入力、1..256混合batchを含む。12種類の4epoch PF比較で保存状態・RNG・graph eventsも一致した。数値許容差を追加していない。

CUDA memcheck/racecheck/synccheckは各50subset PASS、error/hazard/leak0。既存pytest設定warningが各suite1件。全warmで凍結GPU参照の135診断値・status・score/input bitsをdriver内で照合し、全timedの返却hashも一致。raw trace全件のsentinel・訪問順・単調性・区間和・metadata・n11経路・集計を独立auditで再照合した。warmの完全診断hash自体はraw JSONLに保存しておらず、その一致は凍結driverのassertと成功終了で確認した。

新たな400状態のfield再走やfullroute/multiseed精度評価、allocation/async障害注入は行っていない。ここでの実走入力は既存の保存score入力で、全軌跡の新たな速度・精度保証ではない。元NumPy/native差とscalar NaN codecの既存の限界を引き継ぐ。

## 実装と保存

元の数値prefix、CPU score body、同期修正版kernel/headerはliteral保持。計測helperから時刻・metadata・境界呼び出しだけを除くと元helperに一致する。凍結kernelの18336命令・64regs・shared30904B・stack32B・spill0は同一DLLでも一致。計測kernelは18816命令、同期対照18528命令、両方72regs・shared31288B・stack32B・spill0。raw計測kernelは25360命令・96regs・shared30984B・stack0・spill0。静的PE検査であり、実行/JITコードや動的使用率の測定ではない。

syncwarpの上記個数はsourceの呼び出し数。コンパイラは冗長な同期を変換できるため動的な同期回数・時間ではない。追加の静的opcode検査ではWARPSYNCは同期修正版5、時刻無効の対照5、計測版7箇所、SR_CLOCKLO付きCS2Rはそれぞれ0/0/18箇所だった。「区切り同期だけ」の名称はsource上の対照を指し、生成コードで追加の同期だけが揃った比較ではない。markers対比にはコンパイラによる同期整理・命令配置・buffer処理の差も含み、純粋なclock読み出しや同期のコストとは解釈しない。最初の検査は別表記の同期と全CS2Rを検索していたため、別のchecked scriptで照合し、その初回値を同期・時刻読み出し回数の根拠にはしていない。

5timing contextの従来numeric stagingはhost/device各5716760B、copy/barrier/traceの3contextにtrace各196608Bを追加して合計各6306584B。raw test最大temporary device1220608B、driver/heapピーク未計測。全bufferは上限と解放経路を持つ。scoreからraw probeは呼ばない。

36製品source・5installed PE・PF build PE・過去74reports/36archives5974filesを維持。最初の準備scriptは文字列照合で停止し、数値sourceのbuild前に別のchecked準備scriptで修正した。失敗した準備scriptも保存し、実行済みの数値source/test/helperを変更せず、完成した計測をやり直していない。

[結果](../benchmarks/rbpf_lu_phases_20261017.json)、[完了照合](../benchmarks/rbpf_lu_phases_completion_check_20261017.json)、[driver](../benchmarks/bench_rbpf_lu_phases.py)、[CUDA helper](../benchmarks/probes/rbpf_lu_phases/lu_phases.cuh)、[tests](../tests/test_rbpf_lu_phases.py)。
