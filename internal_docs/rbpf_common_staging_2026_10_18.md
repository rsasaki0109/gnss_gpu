# 共通候補の純粋な登録hitをまとめる (D273, 2026-10-18)

採用を見送る。完全PF3840timed/256warmでCPU主gate 0/8、GPU同backend 0/8、GPU joint 0/8。標準条件0はCPU/GPUともFAILであり、事前条件に従い追加fieldは0状態。製品/既定は維持する。機能47tests・CPU ASan35subsetはPASSし、fixture内の全PF状態・RNG・feedback・従来metadataは同backend元処理とbits一致した。これを実データ測位精度や全RTK速度の確認と扱わない。

## 変更する範囲

D236の完全PFをbaselineに、consensus/adopt/spawnという共通suffixの純粋なmemo hitだけをC API内でまとめる。D244の元CPPをそのままincludeし、literal登録とper-hit融合を同じPEの対照として使う。全粒子/全候補/候補順/first-seen Hとscore index/元のlogical canonical requestsを保持する。各粒子のrecordは新しく作り、確率templateは共有しない。weight/choice/logw、粒子順の全RNG draw、H copy/commit、conditional/FIX/feedback/resamplingは元のcodeを維持する。

共通候補の登録が全て純粋な既存fusion hitになったblockの後だけ、prior/index/H/tagをcall内へ保持する。初回のsource callbackで候補listが増える間は登録を元通り実行する。再利用前にnested整数dictのversion、候補順とlist長、cfg/class/instance dict、memoのH自身のcanonical keyと登録index、型/元関数/bindingsを確認する。first-seen登録keyと、等値でも別objectのmemo keyを混同しない。変更/未対応は元登録へ戻る。

## 試験と修正履歴

元のcallback/hash/equality/例外/巨大整数/動的list/参照解放と、CPU/GPU/scalar×2seed×2cluster×4epochの元/同PE対照/候補の完全PF状態を確認した38件に、再利用後のnested/root H・候補list・prior・cfg/context dict置換・custom counter・callback例外の9件を追加し、計47件PASS。再利用を実際に通ったことも必須assertした。CPU ASanは26+9の35subset PASS、受理logにASan診断なし。35件を別の機能件数へ加えない。GPU数値sourceは同一で、新しいCUDA checkerは行っていない。

初回のtest起動はfield環境のthreadpoolctl不足で収集時に止まり、別test moduleを元のNumPy2.3.4環境で使った。初版nativeは組込み型のnull tp_dict参照で停止した。PyType_GetDictへ変えた版は25pass/13経路未使用FAIL、全状態比較は一致した。各Hのmemo keyを保持する版は37pass/1FAILで、初回callback中に候補が増えた際のprior差0.0011111111111111113を検出した。最終warm版はcallbackのない純粋hit blockからだけ保持し、このnegativeを同じassertのまま解決した。全初版source/PE/test/logを残す。速度測定は最終版だけで、完了した測定の取り直しはない。

debug capture buildのcompiler検出失敗、benchmark準備のtext undo重複によるassert失敗も数値実行前の履歴として残す。準備修正は別scriptで行った。試験済みguard testのfixture import/引数名によるF811だけをfileとSHA付きで除外し、数値sourceを変更せず残りのRuff/compile/EOL/diff検査を行う。CPython内部version fieldの既存C4996警告も残す。

## 完全PFの速度

8条件、CPU/GPU各元・同じ新callerのD244対照・候補と、CPU/GPU元A/Aの8mode。4初期RNG seed×5round×3group。全3840timed/256warmを保持し、4stepのinteger-ns合計→3group中央値→5round中央値で比較する。元と対照の両方へ10%以上、4/5round正、対応A/Aのsigned中央値abs5%未満を要求する。GPU jointは候補CPUに対するGPU短縮と両A/Aも必要。表は正が短縮。

| particles/unique/rows/groups,cluster,feedback | CPU対元 | CPU対D244 | GPU対元 | GPU対D244 | CPU A/A | GPU A/A | CPU/GPU同backend/joint |
|---|---:|---:|---:|---:|---:|---:|---|
| 0: 64/6/64/4, c0, fb1 | -2.503% | -3.641% | -2.932% | -2.552% | -7.180% | +5.365% | False/False/False |
| 1: 64/32/64/4, c0, fb1 | -4.093% | -12.179% | -5.623% | -12.336% | -4.719% | -1.543% | False/False/False |
| 2: 64/64/64/4, c0, fb1 | -5.640% | -3.834% | -17.505% | -13.529% | +3.177% | +1.296% | False/False/False |
| 3: 128/32/64/4, c0, fb1 | -6.244% | -5.548% | -18.571% | -23.179% | +1.951% | -3.424% | False/False/False |
| 4: 64/6/128/8, c1, fb1 | -1.497% | -9.880% | +3.745% | -5.093% | +4.326% | +11.233% | False/False/False |
| 5: 128/128/128/8, c1, fb1 | -10.072% | -7.081% | -6.053% | -7.024% | +0.443% | -0.946% | False/False/False |
| 6: 64/6/64/4, c0, fb0 | -4.556% | -7.270% | +0.613% | -2.348% | +2.660% | -5.895% | False/False/False |
| 7: 64/6/64/4, c1, fb1 | +1.182% | -4.664% | +0.650% | -1.490% | -1.831% | -6.422% | False/False/False |

guard/型・version検査/metadata/temporary allocations/scoring/全転送と完了待ち/元choice/全PF tailをtimerへ含める。restore/deepcopy/RNG生成/callback再bind/hash/検証/JSON/gzipは外。許可CPU1本/BLAS1/normal priority/非占有、計算は直列実行。状態の全warm decode、全4096rawの順序/件数/hash/counter、paired集計を別auditで再確認する。extract/search/観測時刻/衛星/graph solveはfixture境界である。

追加fieldへ進む条件はCPU主またはGPU同backendが2/8以上、かつそのbackendの標準条件0も通過すること。今回は標準条件が不通過なので実施しない。PFの候補登録回数削減を全体速度・GPU化率へ読み替えない。通常APIの凍結dependency prototypeであり、任意namespace/property改変・同state並行書込み・OOMの途中stateは一般保証しない。scalar NaN payload codecの従来制限は継承する。数値許容差は追加していない。

## 保存

36製品source・5installed PE・PF build PE・106既報・52archive/10558filesを保全する。新しいGPU kernel/buffer/transferは0、既存scorer contextは1、host/device workspaceは各1143352bytes。追加はcall-local suffix record/整数dict参照/version stamp等で、次callへ保持せずheap peakは未測定。既定の変更はない。

[結果](../benchmarks/rbpf_common_staging_20261018.json)、[完了照合](../benchmarks/rbpf_common_staging_completion_check_20261018.json)、[caller](../benchmarks/rbpf_common_staging.py)、[受理CPP](../benchmarks/probes/rbpf_common_staging_warm/staging.cpp)、[suffix](../benchmarks/probes/rbpf_common_staging_warm/suffix.h)、[完全PF](../benchmarks/bench_rbpf_common_staging.py)。
