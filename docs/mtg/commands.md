# HE姿勢推定　当日のコマンド表

**MacとLinuxのリポジトリ直下で操作。** `YOUR_SSH_ALIAS` とパスは各自の値に置き換えます。GPU番号は割当を確認します。

## 1　Macから接続してライブを起動

```bash
ssh_alias=YOUR_SSH_ALIAS
remote_root=/absolute/path/to/he-pose-mtg
.venv/bin/python -m hepose.cli doctor --role client
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$ssh_alias" true
.venv/bin/python -m hepose.cli live --camera 0 \
  --remote "$ssh_alias" --remote-root "$remote_root" --show-fps
```

画面の `q`、または端末の Ctrl+C で終了。出力先と `summary.json` を控えます。

## 2　Linuxで録画を比較

```bash
.venv/bin/python -m hepose.cli doctor --model yolo26
.venv/bin/python -m hepose.cli doctor --model detrpose
root="outputs/mtg-basic-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python -m hepose.cli benchmark --models yolo26,detrpose \
  --input assets/sample/basic.mp4 --output "$root" --device cuda:0
.venv/bin/python -m hepose.cli compare --input-dir "$root" \
  --output "$root/comparison.mp4"
printf '%s\n' "$root"
```

## 3　Macへ持ち帰って再生

```bash
remote_run=outputs/mtg-basic-YYYYMMDD-HHMMSS
scp -r "$ssh_alias:$remote_root/$remote_run" .
open "${remote_run##*/}/comparison.mp4"
```

推定の状態と元フレーム番号を記録。ライブが止まった場合は `run.log` を確認し、録画処理または手元の比較動画に切り替えます。
