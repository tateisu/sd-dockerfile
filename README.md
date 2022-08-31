# コレは何

Windows 11のWSL2でnvidia-docker2でnvidia PyTorchベースのdockerコンテナを作って、
Stable Diffusion (軽量版) を動かします。

- 現時点ではtxt2img しか動作確認してません。

# 動作環境
- Windows 11
- WSL2
- nVidiaのCUDAが使えるビデオカード。VRAM 6GB以上。

# 使い方

## ホストの設定(1)
まずWindows 11 のPCでCUDA Driver、WSL、docker、nvidia-docker2 などを使えるようにします。
- https://lemmy.juggler.jp/post/12830 を参照

## sudoの設定
- https://qiita.com/YuukiMiyoshi/items/eec3c1827cd8356c1def とか

## このリポジトリのclone
- (リポジトリを建ててから書く)

## コンテナのビルド
- PowerShellを起動
- `wsl` を実行。WSLのUbuntuのシェルが起動する。
- `cd <リポジトリをcloneしたフォルダ>` を実行。シェルのワーキングディレクトリが変わる。
- `./buildContainer.pl` を実行。stable-diffusionのリポジトリのcloneと、dockerイメージの作成が行われる。

## モデルデータのダウンロード
StableDiffusionの公式の説明をみて、チェックポイント(ckpt)ファイルをいくつかダウンロードして
`models/`フォルダの下に置いておく。
** １個あれば動く。 **

```
├── README.md
├── buildContainer.pl
├── ddpm.py
├── diffusers_txt2img.py
├── docker-build-context
│   ├── Dockerfile
│   └── requirements.txt
├── launchContainer.sh
├── models
│   ├── sd-v1-1.ckpt <== 追加
│   ├── sd-v1-2.ckpt <== 追加
│   ├── sd-v1-3.ckpt <== 追加
│   └── sd-v1-4.ckpt <== 追加
├── openaimodelSplit.py
├── optimUtils.py
├── optimized_img2img.py
├── optimized_txt2img.py
├── selectModel.pl
├── txt2imgEx.py
└── v1-inference.yaml
```

## コンテナの起動
- PowerShellを起動
- `wsl` を実行。WSLのUbuntuのシェルが起動する。
- `cd <リポジトリをcloneしたフォルダ>` を実行。シェルのワーキングディレクトリが変わる。
- `sudo service docker start` を実行。
- `launchContainer.sh` を実行。dockerコンテナが開始される。
- 説明：
- `host/selectModel.pl host/models/sd-v1-???.ckpt` を実行。モデルデータのファイル名は実際に追加したものから一つを選ぶこと。
- `python host/txt2imgEx.py --repeat 10 --prompt "great valley"` を実行。
- 終わったらPCで`リポジトリをcloneしたフォルダ/outputs` を見ると画像とパラメータ情報のファイルができている。

## 説明
- `launchContainer.sh` でコンテナを起動すると、`host`フォルダがコンテナ外側の、`リポ ジトリをcloneしたフォルダ`にマッピングされる。
- `host/selectModel.pl` は指定したモデルデータのシンボリックリンクを所定の場所に作ったり、出力フォルダがなければ作ったりする。
- `txt2imgEx.py` は繰り返し時にVRAMを一旦開放することで長時間回しても重くならないようにしている。また、生成時のパラメータを別ファイルに出力する。
