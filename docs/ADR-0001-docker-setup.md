# ADR-0001: Docker による開発・評価環境のセットアップ

- ステータス: Accepted
- 決定日: 2026-08-14

## 背景

PARC 2026 予選配布環境を、ホストの Python 環境へ影響を与えずに利用できる状態にする必要があった。
リポジトリには CPU 用の `Dockerfile` と、依存関係、LIBERO-plus、評価用アセットを一括で取得する
`setup.sh` が用意されている。

初回の `docker build -t parc2026 .` は、LIBERO のチェックアウト中に Docker の保存領域が不足し、
`No space left on device` で失敗した。確認時点では、未使用イメージとビルドキャッシュに
再利用可能な領域が残っていた。

## 決定

開発・評価環境は、リポジトリで提供されている手順どおり Docker で構築する。

容量不足の解消では、使用中のコンテナとボリュームを維持し、再取得・再生成できる次のデータだけを削除する。

- 使用中のコンテナから参照されていない Docker イメージ
- 未使用の Docker ビルドキャッシュ

容量確保後に `parc2026` イメージを再ビルドし、コンテナ内でテストと提出テンプレートの動的検証を行う。
セットアップのためのソースコードおよび設定ファイルの変更は行わない。

## 実施内容

以下のコマンドを実行した。

```bash
docker image prune -a -f
docker builder prune -f
docker build -t parc2026 .
docker run --rm parc2026 pytest tests
docker run --rm parc2026 python validate_submission.py submission_template/
```

未使用イメージ約 5.7 GB と、未使用ビルドキャッシュ約 17 GB を削除した。
構築されたイメージは CPU 用の `parc2026` で、実行環境は arm64、イメージサイズは約 13.3 GB となった。

## 検証結果

- Python 3.10.12
- PyTorch 2.11.0+cpu
- MuJoCo 3.7.0
- Track 1 suite の登録成功
- pytest: 50 passed、1 skipped
- `submission_template/` の動的検証: PASS

動的検証では、テンプレートのランダムポリシーが同一 seed と観測に対して異なる action を返す警告が出た。
これはテンプレート実装の性質によるものであり、セットアップ失敗とは扱わない。

## 影響

### 利点

- ホストの Python 環境を変更せず、配布時の依存関係を再現できる。
- テストと提出前検証を同じイメージで実行できる。

### 注意点

- 削除した未使用イメージを他プロジェクトで再度使う場合は、再取得または再ビルドが必要になる。
- ローカルイメージは CPU・arm64 構成であり、本番の GPU・CUDA 13.0 構成とは異なる。
- 今回はリポジトリ内のコードや設定を変更していない。

## 利用方法

対話シェルは次のコマンドで起動する。

```bash
docker run -it --rm parc2026
```
