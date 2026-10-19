# D290補足: Windows専用診断テストの収集範囲 (2026-10-18)

Ubuntuで全testsを収集するCI設定を確認したため、今回のnative例外診断14件をWindows x64でのみ実行するcollection guardを追加した。測位処理・数値helper/test/runner/PEは変更しない。既存のtest collection pluginに正確な3file名だけを追加し、非Windows/x64以外ではskipする。無関係なテストや既存cycle-counter guardは維持する。

新4policy tests PASS。platformをグローバルに偽装せずexplicit booleanを渡し、非対応環境でのskip、Windows x64での実行維持、既存guard維持、prefix単位の広いskipをしないことを検証した。Windows x64でchecked11+親timeout回帰3+policy4の18件が収集されることもcollect-onlyで確認した。Linux上の実行は今回行っていない。

D290の14件と合わせ18unique testsが通った。数値診断の再走、速度gate再判定、製品/精度/実行時既定の変更は0。旧policy bytesと[D290本体](rbpf_crash_isolation_2026_10_18.md)のレポート・archive・最終照合を保持し、更新したhandoff planを別fileで作成した。142既報/70archives/17828filesの既存保全は、直前のfull検証と新616archive bytesの再照合を引き継ぐ。

[結果](../benchmarks/rbpf_crash_isolation_platform_20261018.json)、[完了照合](../benchmarks/rbpf_crash_isolation_platform_completion_check_20261018.json)、[collection policy](../tests/rbpf_gc_clocks_platform.py)、[policy tests](../tests/test_rbpf_windows_debug_platform.py)。
