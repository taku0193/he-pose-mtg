# 共通基盤の実装記録（途中）

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
