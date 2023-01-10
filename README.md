# MiniCTorch_Benchmark

PytorchのプログラムをMiniCTorchを用いて変換して、ベンチマークを取るためのスクリプト

## 準備
MiniCTorch_Benchmarkリポジトリの直下に以下のディレクトリを配置
- src/ (MiniCtorchのリポジトリのsrc をコピー：　https://github.com/kojima-r/MiniCTorch_Prototype/tree/master/src)

- xtensor/　：xtensorのライブラリのディレクトリ
- xtensor-blas/ ：xtensor-blasのライブラリのディレクトリ
- xtl/ : ：xtlのライブラリのディレクトリ

xtensor関連のディレクトリに関しては以下のコマンドでgitからcloneできる。
```
git clone https://github.com/xtensor-stack/xtensor.git
git clone https://github.com/xtensor-stack/xtensor-blas.git
git clone https://github.com/xtensor-stack/xtl.git
```

## ベンチマーク

- ae:　オートエンコーダ
- vae：　変分オートエンコーダ
- bbb: Bayes by back propergation
- mlp: 多層パーセプトロン
