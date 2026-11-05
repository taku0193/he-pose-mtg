# YOLO26遠隔ライブの実装・検証（2026-11-05）

MacのカメラをOpenCVで読み取り、既存のSSH認証でLinuxのYOLO26ワーカーへJPEGを送る。ワーカーごとにLinuxのループバック上で空きポートを使い、OpenSSHの `-W` で接続する。セッション・通信版・アプリ版・ソースコミットを確認してからGPUを初期化する。画像と応答のサイズ・フレーム番号を検証し、取得時の画像にだけ骨格を重ねる。処理中は待機画像を最大1枚に保つ。

`src/hepose/live*.py` にクライアント、通信、取得、ワーカーを分け、CLIと設定例を追加した。既定はログと統計を保存する。録画は `--record` で明示する。SSHの切断時はエラーと取得済み統計を残す。ワーカーには応答やセッション全体の期限があり、ネイティブ処理が止まった場合にも専用プロセスを終了する。

モデルなしのテストでは、通信モジュールがない段階でのRedと実装後のGreenを確認した（`work/tdd/20-live-red.log`、`21-live-green.log`）。パケット長、画像サイズ、期限、フレームの不一致、起動前の版照合、最新フレームの置き換え、切断時のアダプタ解放を検証した。カメラとウィンドウを代用した試験では `q`、1枚の表示、使用資源の解放、既定出力を確認した。全単体テスト99件＋11 subtests、ruff、CLI helpを実行した。

## abertでの実動作

同一Linuxホストのクライアントから既存SSHでabertへ接続した。ワーカーは割当のGPU 0（RTX 3090）、YOLO26入力640、JPEG品質80。入力は[出典を確認した運動動画](../../assets/sample/jumping-jacks-ATTRIBUTION.md)の640×480、25 FPS、375枚。通常再生・負荷試験・障害注入は実装コミット `0af7c37337a9c8260224ece81208e2b095d84c57` で実施した。

| 条件 | 取得 | 結果 | 待機中の置き換え | 送信後に応答なし |
| --- | ---: | ---: | ---: | ---: |
| 25 FPSで全編を再生 | 375 | 369 | 6 | 0 |
| 120 FPSで200枚を再生 | 200 | 59 | 141 | 0 |
| 実行が起動した後に所有SSHトンネルを停止 | 10 | 3 | 6 | 1 |
| 実行が起動した後にCtrl+C（SIGINT） | 10 | 3 | 6 | 1 |

通常再生では結果24.59件/秒。取得直後から描画終了までのp50は33.27 ms、p95は50.89 ms。取得フレーム数と破棄・処理内訳が一致し、受信結果のフレーム番号は重複せず昇順だった。録画MP4も369枚と確認した。切断は終了コード1、Ctrl+Cは130で、どちらも部分統計を保存した。別試験でセッション上限を5秒へ短縮し、処理中に期限で終了して部分統計を保存することも確認した。すべての試験で実行が起動したワーカーのPIDが残っていないことを確認した。

実ファイルはabertの `outputs/live-ssh-final`、`outputs/live-ssh-overload`、`outputs/live-ssh-final-disconnect`、`outputs/live-ssh-final-sigint`、`outputs/live-ssh-expiry`。集計・条件は[実験記録](../experiments/remote-live-2026-11-05.json)に保存した。再現例：

```bash
.venv/bin/python -m hepose.cli live \
  --video assets/sample/jumping-jacks-validation.mp4 --headless --fps 25 \
  --remote taku@abert --remote-root /home/taku/he-pose-mtg \
  --network-note 'Linux client and worker on abert; same-host OpenSSH' \
  --record --output outputs/live-ssh-recheck
```

表示を行わない録画再送であり、表示FPSとアプリ内表示遅延は未測定。同一ホストでの短い実動作確認であり、Macや研究室のネットワーク性能は推定しない。Macのカメラ許可・OpenCV表示・実機のqと切断、および3人分の受入を[READMEの手順](../../README.md#macからyolo26を遠隔実行する)で行う。次の実装工程は同一素材・同一GPUでの3モデルの順次計測と比較動画。
