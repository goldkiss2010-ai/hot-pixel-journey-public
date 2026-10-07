# ホットピクセルの一生

Bayer RAW上の単一ホットサンプルが、demosaic、white balance、color correction matrix、clipを経て最終RGBへ到達するまでを、QuartoとPythonで追跡する実験記事です。

公開記事: https://goldkiss2010-ai.github.io/hot-pixel-journey-public/
解説動画（日本語）: https://youtu.be/g9p0SmkTJ8A

## 内容

このリポジトリには、記事本文と再現用コードを収録しています。

- `index.qmd` — Quarto記事本文
- `src/demosaic.py` — Bilinear / Malvar–He–Cutler demosaic
- `src/experiments.py` — ホットピクセル、step edge、shot/read noiseなどの合成入力
- `src/pipeline.py` — white balance、CCM、clip、各種評価
- `src/diagram_style.py`, `src/diagrams.py` — 記事内の模式図と共通スタイル
- `references.bib` — 参考文献

記事では、単一異常の空間的伝播、負係数を含む線形再構成によるovershoot / undershoot、CFA位相差、欠陥補正の順序、WB / CCM / clip、ノイズ相関、RGB立方体上でのheadroomを同じコードから可視化します。

## 再現

Quarto、Python、`uv` が利用できる環境を想定しています。

```powershell
git clone https://github.com/goldkiss2010-ai/hot-pixel-journey-public.git
cd hot-pixel-journey-public

uv venv
uv pip install -r requirements.txt
quarto preview
```

本稿で使用するCCMは説明用の合成行列です。特定のカメラプロファイルを再現するものではありません。また、欠陥補正も処理順序を観察するための簡単なモデルです。


## ライセンス

コードはMIT License、記事本文・ドキュメント・オリジナル図版はCC BY 4.0です。詳しくは `LICENSE` を参照してください。

## 引用

`CITATION.cff` を収録しています。GitHubの “Cite this repository” から引用情報を取得できます。
