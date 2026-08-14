# LIBERO Spatial 悪化タスク分析

## 10エピソード・複数seed再評価（最新）

2026年8月14日に、BaseとSpatial LoRAを同じ設定で全10 task、各10エピソードへ拡張して再評価した。環境seedは各taskで2026から2035までを使用した。評価器の確率的policy RNGは実行中に連続するため、各行は同じ環境seedの対応比較だが、policy乱数までepisodeごとに独立固定した比較ではない。

| task_id | Base | Spatial LoRA | 差 | Baseのみ成功 | LoRAのみ成功 |
|---:|---:|---:|---:|---:|---:|
| 0 | 9/10 | 9/10 | 0ポイント | 1 | 1 |
| 1 | 10/10 | 9/10 | -10ポイント | 1 | 0 |
| 2 | 8/10 | 9/10 | +10ポイント | 0 | 1 |
| 3 | 8/10 | 6/10 | **-20ポイント** | 4 | 2 |
| 4 | 9/10 | 10/10 | +10ポイント | 0 | 1 |
| 5 | 5/10 | 5/10 | 0ポイント | 2 | 2 |
| 6 | 7/10 | 7/10 | 0ポイント | 1 | 1 |
| 7 | 4/10 | 7/10 | **+30ポイント** | 1 | 4 |
| 8 | 8/10 | 5/10 | **-30ポイント** | 4 | 1 |
| 9 | 10/10 | 10/10 | 0ポイント | 0 | 0 |
| **全体** | **78/100** | **77/100** | **-1ポイント** | **14** | **13** |

対象としていたtask 3・6・7の合計はBase 19/30（63.3%）、Spatial LoRA 20/30（66.7%）で、LoRAが3.3ポイント高かった。したがって、3エピソード評価で見えた「3・6・7がすべて悪化」という仮説は再現しなかった。個別にはtask 3の悪化傾向が残り、task 6は同率、task 7は改善した。代わりにtask 8が8/10から5/10へ低下した。

全体成功率の95% Wilson区間はBase 68.9–85.0%、Spatial LoRA 67.8–84.2%で大きく重なる。各taskも10試行では区間が広く、task 3・7・8の20–30ポイント差を確定的な転移と判断するには不足している。次は20 seed以上へ拡張し、episodeごとにpolicy RNGも同じseedへリセットした対応評価を行う。

実測JSON:

- Base: `/Users/takemo/PARC2026_outputs/20260814_121853/eval_10ep_seed2026/base/eval_info.json`
- Spatial LoRA: `/Users/takemo/PARC2026_outputs/20260814_121853/eval_10ep_seed2026/spatial_lora/eval_info.json`

この結果から、旧3エピソード結果だけを根拠にtask 6・7へ一律増量する案は採用しない。暫定的にはtask 3と新たに低下したtask 8を分析・corrective data候補とし、task 6・7は能力維持用rehearsalとして残す。task 8は失敗段階をまだ分類していないため、追加配分を確定する前に失敗動画と状態遷移を取得する。

## 初回3エピソード評価でわかった事実

Mac（MPS）上で同一条件（seed 2026、各3エピソード）を比較したところ、Baseはタスク3・6・7ですべて3/3成功、Spatial LoRAは各2/3成功でした。

| task_id | タスク | Base | Spatial LoRA | LoRA失敗episode / seed |
|---:|---|---:|---:|---:|
| 3 | ramekin横のblack bowlをplateへ配置 | 3/3 | 2/3 | 2 / 2028 |
| 6 | stove上のblack bowlをplateへ配置 | 3/3 | 2/3 | 2 / 2028 |
| 7 | wooden cabinet上のblack bowlをplateへ配置 | 3/3 | 2/3 | 0 / 2026 |

各タスクの差は1試行だけなので、現段階では「負の転移が確定した」とは言わず、負の転移の疑いとして扱う。再現計測では元評価と9件すべて同じ成否になり、少なくとも今回の失敗は再現可能であることを確認した。負の転移の有無を統計的に判断するには、対策前後を少なくとも20 seedで対応比較する。

## 再現・保存方法

`examples/analyze_libero_negative_transfer.py` は、元評価と同じpolicy RNG系列を保つためtask 0から7までを順に再生する。対象task 3・6・7について、失敗エピソードだけを次の形式で保存する。

- `task_<id>_episode_<index>_seed_<seed>.mp4`: agentviewの失敗動画
- `task_<id>_episode_<index>_seed_<seed>.jsonl`: step、action、reward、terminated、truncated、成功判定、手先・gripper・joint・source/target pose
- `analysis_summary.json`: 成否、ファイルパス、自動分類、判定根拠

実測成果物は `/Users/takemo/PARC2026_outputs/20260814_121853/failure_analysis/spatial_lora_replay_v3/` に保存した。280 stepの上限に達した失敗では、JSONL最終行を `done=true`、`truncated=true`、`termination_reason=max_episode_steps` として明示する。

再実行コマンド:

```bash
MUJOCO_GL=glfw PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTHONPATH=/Users/takemo/Library/Caches/PARC2026/smolvla/LIBERO-plus:/Users/takemo/Library/Caches/PARC2026/smolvla/lerobot-v0.6.0/src \
python examples/analyze_libero_negative_transfer.py \
  --policy-path /Users/takemo/PARC2026_outputs/20260814_121853/smolvla_libero_plus_spatial_lora_merged \
  --output-dir /Users/takemo/PARC2026_outputs/20260814_121853/failure_analysis/spatial_lora_replay_v3 \
  --target-task-ids 3 6 7 --episodes 3 --seed 2026 --device mps
```

分類規則は以下の順序で適用する。

1. 手先と対象bowlの最短距離が10cmを超える: **到達失敗**
2. 10cm圏内へ到達したが、接触把持がなく、初期位置から3cm以上持ち上がらない: **把持失敗**
3. 接触把持または3cm以上の持ち上げはあるが、`On(bowl, plate)`を満たさない: **配置失敗**

閾値付近のエピソードは動画で目視確認し、自動分類を確定ラベルとして扱わない。

## 失敗分類の実測結果

| task_id | 分類 | 最短手先―対象距離 | 最大持ち上げ | 接触把持 | 動画・状態遷移からの所見 |
|---:|---|---:|---:|---|---|
| 3 | **把持失敗** | 5.77 cm（step 55） | 0.20 cm | なし | ramekin横の対象付近まで接近したが、把持せず離脱。最終の対象―plate距離は25.54 cmで、搬送へ移れていない |
| 6 | **把持失敗** | 7.19 cm（step 48） | 0.16 cm | なし | stove上の対象へ接近したが、対象を持たないまま後半はplate側へ移動。最終の対象―plate距離は48.90 cm |
| 7 | **配置失敗** | 6.22 cm | 13.57 cm | step 65–129 | cabinet上で把持・搬送・解放まで完了。対象―plate距離は最小4.02 cm、最終4.02 cmだが、plate縁寄りで `On(bowl, plate)` を満たさない |

3件とも手先は対象の10 cm圏内へ入っているため、到達失敗はなかった。task 3・6は、対象位置や高さが異なっても「接近後に把持を成立させない」という共通モードである。task 7は把持能力を維持している一方、リリース位置の約4 cmの横ずれを最後まで補正できていない。

task 3・6ではgripper action自体は閉方向へ切り替わるため、単純な「閉じ忘れ」よりも、閉鎖時の手先中心・高さ・姿勢のずれが疑われる。task 6はstoveとの干渉余裕も小さい。task 7では搬送後に対象をplate近傍へ置けているため、plate中心へ寄せる終端軌道、接触後の安定待ち、必要なら再把持して補正する例が不足している可能性が高い。

## 初回3エピソードに基づくデータ配分案（再検討対象）

以下は初回3エピソードから作成した案である。10エピソード再評価ではtask 6の悪化が消え、task 7は改善したため、この比率をそのまま次回学習へ適用しない。task 8の失敗分類後に改訂する。

| データ群 | batch比率 | 群内の重点 | 狙い |
|---|---:|---|---|
| task 3 corrective / Base成功軌跡 | 15% | approach 25%、grasp 60%、place 15% | ramekin横での横ずれを補正し、把持成立まで学習 |
| task 6 corrective / Base成功軌跡 | 15% | approach 25%、grasp 60%、place 15% | stove高さ・接触面を条件化した把持を回復 |
| task 7 corrective / Base成功軌跡 | 10% | grasp 15%、transport 25%、place/release 60% | plate中心への終端精度と安定配置を改善 |
| 残り7 Spatial task | 40% | taskごとに均等 | 現行LoRAの全体性能を維持 |
| 汎用pick-and-place rehearsal | 20% | 高さと配置先を層化 | catastrophic forgettingと3タスクへの過適合を抑制 |

corrective軌跡は失敗動画を正例として再利用せず、同じ初期状態から人手またはBase policyで成功させた軌跡を収集する。特に `task 3/seed 2028`、`task 6/seed 2028`、`task 7/seed 2026` を優先し、その近傍へ対象位置を摂動した例を追加する。各悪化task内では同じ成功軌跡を単純複製せず、初期高さ、横方向、接近角度、背景を層化して均等sampleする。

学習安定化の初期値はLoRAのlearning rateを現状の1/2、悪化3タスクのsample weight上限を3.0とし、10%のBase rehearsalを常に含める。全体成功率が落ちる場合は、追加stepを増やす前に悪化3タスク比率を40%から30%へ下げる。

## 改善実験

1. **配分のみのablation**: 上記40/40/20配分で学習し、現行LoRAと比較する。
2. **把持フェーズ強化**: task 3・6について、gripper閉鎖前後±15 stepを重点sampleし、手先―対象相対poseをばらつかせる。
3. **配置フェーズ強化**: task 7について、plate中心から半径0–2 cmを成功目標とする軌跡、接触後5–10 step静止する軌跡、縁への配置から再補正する軌跡を加える。
4. **段階別評価**: 成功率だけでなく、到達率、把持率、搬送率、配置述語達成率をseedごとに保存し、改善が次段階の失敗へ移っただけでないことを確認する。

## 採否基準

調整案は、20 seed以上の対応比較で次を同時に満たす場合だけ採用する。

1. task 3・6・7の平均成功率がBase比で非劣性（許容差5ポイント以内）
2. 全10 task平均が現行LoRAの83.3%以上
3. task 3・6の把持率、task 7の配置述語達成率がそれぞれ現行LoRA以上
4. 到達・把持・配置の別段階へ失敗を移していない
