# HE姿勢推定　当日のコマンド表

**Mac → Linux → Macの順で操作。** 下のSSH名とパスはabertの例です。各自の値に置き換えます。GPUは `hepose.local.toml` の `device` と `gpu_uuid` で指定します。

## 1　Macから接続してライブを起動

Mac側のリポジトリへ移動してから実行します。

```bash
ssh_alias=taku@abert
remote_root=/home/taku/he-pose-mtg
.venv/bin/python -m hepose.cli doctor --role client &&
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$ssh_alias" true &&
.venv/bin/python -m hepose.cli live --camera 0 \
  --remote "$ssh_alias" --remote-root "$remote_root" --show-fps
```

画面の `q`、または端末の Ctrl+C で終了。出力先と `summary.json` を控えます。

## 2　Linuxで録画を比較

ライブを閉じたら、同じMacのターミナルで `ssh "$ssh_alias"` を実行します。接続後、Linuxで次を実行します。

```bash
cd /home/taku/he-pose-mtg &&
root="outputs/mtg-basic-$(date +%Y%m%d-%H%M%S)" &&
.venv/bin/python -m hepose.cli doctor --model yolo26 &&
.venv/bin/python -m hepose.cli doctor --model detrpose &&
.venv/bin/python -m hepose.cli benchmark --models yolo26,detrpose \
  --input assets/sample/basic.mp4 --output "$root" &&
.venv/bin/python -m hepose.cli compare --input-dir "$root" \
  --output "$root/comparison.mp4" &&
printf '%s\n' "$root"
```

成功したら最後に表示されたフォルダ名を控え、`exit` でMacに戻ります。失敗時は端末のエラーを確認します。

## 3　Macへ持ち帰って再生

`run_name` に、先ほどの出力先から `outputs/` を除いたフォルダ名を入れます。

```bash
run_name=mtg-basic-YYYYMMDD-HHMMSS
scp -r "$ssh_alias:$remote_root/outputs/$run_name" . &&
open "$run_name/comparison.mp4"
```

推定の状態と元フレーム番号を記録。ライブが止まった場合は `run.log` を確認し、録画処理または手元の比較動画に切り替えます。
