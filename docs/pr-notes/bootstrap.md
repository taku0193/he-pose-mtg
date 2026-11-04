# 共通基盤の実装記録

日付：2026-11-04。対象ブランチ：chore/bootstrap。仕様：v1.3。

## 最初の区切り

- Pythonパッケージ、共通CLI help、client／workerに分けたdoctorの共通部分。
- スキーマ1.0、COCO17の名前対応、欠測null、クロップ座標復元、人物・フレーム・時間窓の契約。
- モデル別環境のためのアダプタインターフェース。実モデルアダプタは未実装。
- 計時・集計、JSONL／CSV、環境・要約・ログ、既存出力の保護、失敗・中断・0件成功の区別。
- abertのGPU 0使用許可と、ユーザー提示の端末出力を環境記録へ反映。

## 実行したTDD

追加パッケージやモデルを使わず、PYTHONPATH=src python3 -m unittest discover -s tests -vで実行。

| 区切り | Red | Green |
| --- | --- | --- |
| CLI・座標・時間窓・統計 | hepose.cli／schema未実装を原因とする失敗を確認 | 13件成功 |
| 結果ファイル保存 | hepose.results未実装によるImportErrorを確認 | 合計18件成功 |
| 推論対象0件の誤った成功判定 | 空の実行をsuccessとするテストが失敗 | 修正後、合計19件成功 |

生ログは作業領域work/tdd/01-contracts-red.log・01-contracts-green.log・02-results-red.log・02-results-green.log・03-empty-run-red.log・03-empty-run-green.logに保存。作業領域のパスは実装を実行した場所に対応する。CLIと全サブコマンドのhelp、compileallによる構文確認も実行済み。

## 実動作と制限

CPUで上記の単体テストを実行した。入力はテスト用データであり、姿勢推定の実測・実動画結果ではない。doctorのモデル状態とrun等の実行は未実装を明示し、成功として扱わない。

ruffは未導入・未実行。動画I/O・骨格描画・実行パイプライン・依存版固定・CIは未実装。GPU推論・モデル重み・Macカメラ・SSH遠隔ライブ・性能測定は未実施。共通基盤全体を完了とせず、mainへコードを統合する条件も未達。

## 次の作業

動画I/Oと描画をテストから実装し、モデル不要のアダプタをテスト内で注入する。必要な依存を開発環境に導入・固定し、ruffと単体テスト・CLI helpを確認してから共通基盤を統合する。

## 動画パイプラインを含む共通基盤の検証

動画I/O・骨格描画・共通6出力、時間窓の境界、設定とモデル別Pythonの振り分けを実装した。Linuxの.venvで50件のCPUテスト、ruff check、ruff format --check、全サブコマンドのhelpを確認した。以前の「実動作と制限」「次の作業」は初期commit a207ee2時点の記録であり、今回の残作業を表さない。

| 区切り | Redの確認内容 | Green |
| --- | --- | --- |
| 動画I/O・描画・パイプライン | render／pipelineモジュール未実装のImportError | 追加14件成功 |
| ローカル設定・環境振り分け | configモジュール未実装のImportError | 追加5件成功 |
| 例外処理 | 読込失敗時のログ欠落、子プロセスの不正JSON、不正deviceの終了コード | 修正して合計45件成功 |
| 仮想環境 | Pythonリンクの解決で別環境が同じ環境と誤判定される2件の失敗 | 合計47件成功 |
| クライアントのGUI | headless OpenCVが表示可能か区別できない失敗 | 合計48件成功 |
| ワーカー起動と中断 | 不在Python時のログ欠落、SIGINTが130にならない失敗 | 合計50件成功 |
| デバイスの確認元 | 要求GPUと実使用デバイスを分けて保存できない失敗 | 既存テストへ回帰条件を追加して成功 |

新しい生ログはwork/tdd/04-video-pipeline-{red,green}.logから10-device-provenance-{red,green}.logに保存した。途中のコード編集で発生した構文・lintエラーも修正し、最終確認は別途実行する。

成功時は同じフレーム数・FPSのMP4を再読込みして確認し、JSONL／CSV・要約・環境・ログを保存する。失敗や中断時は所有する動画I/Oとアダプタを解放する。入力SHA256、Gitコミット・dirty状態、依存版、要求デバイスを保存し、実使用デバイス・モデル時間が不明ならnullを維持する。

uv.lockに依存版と配布物のハッシュを固定し、ビルド用setuptoolsも版固定した。GitHub ActionsにLinux／macOSのモデル不要テストとlint・helpを定義した。CI実行結果はGitHub側で確認し、各参加者の実機試験と区別する。

## 残る必須機能

3モデルの実アダプタと公式重み、GPU推論、MacカメラとSSHライブ、ベンチマーク・比較動画・実素材・MTGガイドは未実装または未実証。初期の動画出力は固定FPS・偶数サイズのMP4で、可変FPSと全ての未知コーデックの破損検出は対象外。

テスト用の画像・アダプタを実際の姿勢推定結果や性能として扱わない。共通基盤の検証完了と、必須モデルの実装完了を区別する。

## 最終確認

コードcommit d5fc143で、ローカルの50件（追加のsubtest 11件を含む出力）、ruff check／format、依存ロックと全CLI helpを確認した。最終生ログはwork/tdd/final-checks.logに保存した。

GitHub Actionsの[Model-free checks](https://github.com/taku0193/he-pose-mtg/actions/runs/41762661596)も同じコードcommitで成功を確認した。共通基盤をmainへ統合し、次はfeat/yolo26-poseで公式モデル対応へ進む。実モデルの性能やMacカメラ・遠隔ライブの実機検証は未実施のまま。
