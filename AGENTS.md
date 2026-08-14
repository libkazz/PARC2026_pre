# Repository Guidelines

## プロジェクト構成とモジュール

このリポジトリは PARC 2026 予選用の評価キットです。評価処理は `pipeline/` にあり、`pipeline/__main__.py` がコマンドラインの入口です。Track 1 のタスク登録とメタデータは `compe/t1/`、参加者向けポリシーサーバーのひな形は `submission_template/`、学習例は `examples/` に配置します。ルートの `evaluate.py` は一括採点、`validate_submission.py` は提出物検査を担当します。単体テストは `tests/` に置き、対象のエントリーポイントに対応させてください。

## ビルド・テスト・開発コマンド

- `bash setup.sh`: Python 3.10 の仮想環境を作り、固定済み依存関係と LIBERO アセットを取得します。初回のみ実行し、10〜20分程度を見込んでください。
- `source env.sh`: 作成済み環境と必要なパスをシェルごとに有効化します。
- `pytest tests`: 単体テストとスモークテストをすべて実行します。
- `python validate_submission.py submission_template/`: 必須ファイル、エンドポイント、応答形式、レイテンシを検査します。
- `python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 10`: 起動中のポリシーサーバーを短時間で評価します。
- `docker build -t parc2026 .`: CPU 用の参照環境をビルドします。

## コーディング規約と命名

Python 3.10 を対象とし、インデントは4スペースにします。既存の PEP 8 風の書式に合わせ、モジュール・関数・変数は `snake_case`、クラスは `PascalCase`、定数は `UPPER_CASE` を使います。公開関数や意図が分かりにくいデータ構造には型ヒントを付けてください。ポリシーの HTTP・シリアライズ処理は `submission_template/policy_server.py` との互換性を維持します。formatter や linter の設定は未配置のため、周辺コードのスタイルを守り、変更範囲を絞ってください。

## テスト方針

テストには `pytest` を使います。ファイル名は `test_<対象>.py`、関数名は `test_<振る舞い>()` としてください。不具合修正には回帰テストを追加し、特に検証エラー、タスク絞り込み、採点、アーカイブ安全性の境界を確認します。ファイル操作には `tmp_path`、入力差分には parametrization を優先します。動的スモークテストは依存不足で skip される場合があるため、提出前に `setup.sh` で作成した環境でも実行してください。

## コミットとプルリクエスト

直近の履歴では、`README:` のような対象名を付けた簡潔な日本語要約が使われています。新規コミットは日本語の Conventional Commits とし、例は `fix: 不正な提出ZIPの検証を強化` です。1コミットを1つの論理的意図に限定してください。PR には動機、変更した振る舞い、確認コマンドと結果、関連 Issue を記載します。Notebook や文書で表示結果が重要な場合のみスクリーンショットを添付します。Draft で作成し、CI とレビュー指摘への対応後に Ready for review へ変更してください。

## セキュリティと設定上の注意

生成された `venv/`、`env.sh`、取得済み LIBERO ディレクトリ、モデル重み、提出物、評価結果はコミットしないでください。ZIP 展開、外部依存元、リクエストのタイムアウト、パス処理はセキュリティ上重要です。`validate_submission.py` の検査を維持してください。
