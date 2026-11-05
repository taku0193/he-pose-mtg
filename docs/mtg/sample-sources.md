# MTG素材の出典と再現

比較動画には元画像、推定した骨格、モデル名、フレーム番号と時刻を重ねています。下記の元動画とその比較・録画再送の派生動画は **CC BY-SA 4.0** です。出典、著者、加工内容を添えて使います。 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

| ファイル | 原題・著者 | 抜粋と加工 |
| --- | --- | --- |
| `basic.mp4` | Squat and Frontal Raise · Taco fleur | 元動画の2秒から12秒間。25 FPS、H.264、音声を除去 |
| `fast.mp4` | Jumping jacks and burpees · Taco fleur | 元動画の冒頭15秒。既存の `jumping-jacks-validation.mp4` と同一バイト列 |
| `multi.mp4` | Salsa Dancing Fundamentals · Floresaj | 元動画の43秒から12秒間。縦横比を保って縮小し上下に余白、25 FPS、H.264、音声を除去 |

- `basic` の作者・公開条件。 [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Squat_and_Frontal_Raise.webm)
- `fast` の作者・公開条件。 [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm)
- `multi` の作者・公開条件。 [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Salsa_Dancing_Fundamentals.webm)

| 配布ファイル | サイズ・FPS・枚数 | SHA256 |
| --- | --- | --- |
| basic.mp4 | 640×480・25・300 | `51eb714f88e927f8b716badb895867144772a87d7ef3b2db76f4b1b4b4919135` |
| fast.mp4 | 640×480・25・375 | `67d48a46e86fe66e8d096761f0c23098097855b2d04a94a8d88bf14d00d16642` |
| multi.mp4 | 640×480・25・300 | `03d9fb663e064a9c81701c0197ef946f18673bae6709664c8f31924d23de38df` |

参加者は配布セット内の `clips/` からLinuxリポジトリの `assets/sample/` へ配置します。Linuxで `sha256sum assets/sample/{basic,fast,multi}.mp4`、Macで `shasum -a 256 clips/{basic,fast,multi}.mp4` を実行すると、表の配布版と照合できます。

## 元動画から作り直す場合

Linuxのリポジトリ直下で実行します。元ファイルのSHA256、変換時のFFmpeg版と実行条件は記録にあります。エンコーダーの版が違う場合、生成されたバイト列が異なることがあるので、MTGでは配布済みの同一ファイルを全員で使います。 [素材の機械可読記録](../validation-assets/mtg-samples.json)

```bash
mkdir -p work/mtg-assets assets/sample
curl -fL https://upload.wikimedia.org/wikipedia/commons/7/7f/Squat_and_Frontal_Raise.webm -o work/mtg-assets/squat.webm
curl -fL https://upload.wikimedia.org/wikipedia/commons/e/ed/Salsa_Dancing_Fundamentals.webm -o work/mtg-assets/salsa.webm
sha256sum work/mtg-assets/{squat,salsa}.webm
filters='scale=640:480:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25'
ffmpeg -v error -nostdin -n -ss 2 -i work/mtg-assets/squat.webm -t 12 -an -vf "$filters" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  -metadata 'comment=Squat and Frontal Raise by Taco fleur | CC BY-SA 4.0 | excerpt 2-14 s, CFR25, no audio' assets/sample/basic.mp4
ffmpeg -v error -nostdin -n -ss 43 -i work/mtg-assets/salsa.webm -t 12 -an -vf "$filters" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  -metadata 'comment=Salsa Dancing Fundamentals by Floresaj | CC BY-SA 4.0 | excerpt 43-55 s, CFR25, no audio' assets/sample/multi.mp4
python3 scripts/prepare_validation_video.py
cp -n assets/sample/jumping-jacks-validation.mp4 assets/sample/fast.mp4
```

すでに検証素材がある場合は `prepare_validation_video.py` の実行を省略します。共有セットには元の全編WebMを同梱せず、上の切り出し版を添えています。
