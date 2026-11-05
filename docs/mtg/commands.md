# HE姿勢推定　当日のコマンド表

**abertのタブは `ssh abert` で接続済みの状態で使います。** カメラ表示とコピーはMacの別タブで行います。GPUはabertの `hepose.local.toml` で設定します。

## 1　Macの別タブでライブを起動

Mac側のリポジトリへ移動してから実行します。

```bash
.venv/bin/python -m hepose.cli doctor --role client &&
.venv/bin/python -m hepose.cli live --camera 0 \
  --remote abert --remote-root /home/taku/he-pose-mtg --show-fps
```

画面の `q`、または端末の Ctrl+C で終了。出力先と `summary.json` を控えます。

## 2　接続済みのabertで録画を比較

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

最後に表示されたフォルダ名を控え、Macの別タブへ切り替えます。失敗時は端末のエラーを確認します。

## 3　Macの別タブで結果をコピーして再生

`run_name` に、出力先から `outputs/` を除いたフォルダ名を入れます。

```bash
run_name=mtg-basic-YYYYMMDD-HHMMSS
mkdir -p outputs &&
scp -r "abert:/home/taku/he-pose-mtg/outputs/$run_name" outputs/ &&
open "outputs/$run_name/comparison.mp4"
```

推定の状態と元フレーム番号を記録。ライブが止まった場合は `run.log` を確認し、録画処理または手元の比較動画に切り替えます。
