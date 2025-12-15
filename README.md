# Stormworks Custom Engine Sound Generator

Stormworks の Component MOD 用に、カスタムエンジン音源を含む `.bin` ファイルを自動生成する Python GUI ツールです。

## 機能

1. **動画・音声の読み込み**  
   エンジンの回転域に応じた4種類のエンジン音源（L1～L4）を読み込みます。MP4, WAV, OGG など対応形式ならOK。

2. **シームレスループ生成（Overlap-Add 方式）**  
   指定された秒数範囲を切り出し、先頭と末尾を 0.1 秒のリニアクロスフェードで接合。クリック音のない滑らかなループを生成します。RMS 正規化により音量の急変も防止。

3. **OGG Vorbis に直接変換**  
   中間ファイル不要。PCM バッファから直接 OGG へエンコード（品質 0～10、デフォルト 3）。

4. **XML テンプレートの自動生成**  
   エンジン名を反映した `modular_engine_crankshaft.xml` を作成します。

5. **component_mod_compiler で `.bin` 生成**  
   最終的に Stormworks が読める Component MOD バイナリを生成します。

## 必要な環境

- **Python 3.11 以上**
- **FFmpeg** がインストール済み（PATH に登録されていること）
- **PySide6, librosa, soundfile, audioread, numpy** 等（自動インストール）

## 起動方法

```bash
# Windows の場合（推奨）
run_tool.bat

# または直接実行
python -m engine_tool.run_tool
```

**初回起動時の自動セットアップ**  
1. `.venv` 仮想環境を自動作成
2. セットアップウィンドウで依存パッケージをインストール（進捗表示）
3. その後、GUI が起動

2 回目以降は依存パッケージがキャッシュされるため、すぐに GUI が表示されます。

## 使い方

### 1. エンジン名の入力
   任意の名前を指定（例：「LFA」「2JZ」「RB26」）

### 2. ソースファイルの選択
   **「Multi-select sources…」** ボタンで 1～4 個のファイルを一括選択、または個別の「Browse…」で指定

   - 複数選択した場合、先頭から L1 → L2 → L3 → L4 に自動割り当て
   - 個別編集も可能

### 3. ループ秒数を設定
   各ステージ（L1～L4）ごとに：
   - **Start (s)** → ループ区間の開始秒（デフォルト 1.0 s）
   - **End (s)** → ループ区間の終了秒（デフォルト 3.0 s）

   > 💡 **ヒント**: エンジン音が安定している区間を選ぶと良い結果が得られます。

### 4. パラメータ調整（オプション）
   - **Vorbis quality** → OGG 圧縮品質（0～10、デフォルト 3、高いほど高品質・ファイル大）

### 5. 「Generate」をクリック
   自動処理開始。ログパネルに進捗が表示されます。

## 出力ファイル

実行完了後、`results/` 直下に以下のフォルダが作成されます：

```
results/
└── <engine-name>/
    ├── [<ENGINE-NAME>]modular_engine_crankshaft.xml
    ├── <engine-name>-L1.ogg
    ├── <engine-name>-L1.wav          ← デバッグ用（波形確認）
    ├── <engine-name>-L2.ogg
    ├── <engine-name>-L2.wav
    ├── <engine-name>-L3.ogg
    ├── <engine-name>-L3.wav
    ├── <engine-name>-L4.ogg
    └── <engine-name>-L4.wav
```

- XML ファイルと OGG ファイルを `component_mod_compiler.com` でコンパイルして `.bin` を生成
- WAV ファイルは Audacity 等で波形を確認する用途に使用可能

## ループ処理の仕組み

本ツールは **Overlap-Add 方式** でシームレスループを生成します：

1. 指定範囲の音声を切り出し
2. 先頭 0.1 秒（head）と末尾 0.1 秒（tail）を抽出
3. tail → head へリニアクロスフェードで滑らかに接続
4. クロスフェード区間の RMS を調整し、音量の急変を防止
5. ループ長は元の長さから 0.1 秒短くなる

この方式により、ゼロクロス検出や複雑な位相調整なしで、クリック音のないループを実現しています。

## トラブルシューティング

| 状況 | 対応 |
|------|------|
| GUI が起動しない | FFmpeg が PATH に無いか確認。`ffmpeg -version` で確認してください |
| コンパイル失敗 | ログで「Compiler finished」以降の出力を確認。XML/OGG に異常がないか見直してください |
| ループにクリック音がある | WAV ファイルを Audacity で開き、先頭付近の波形を確認してください |
| 音量が不安定 | ソース音声の安定した区間を選び直してください |

## 開発環境

- **Language**: Python 3.11+
- **GUI**: PySide6
- **Audio Processing**: librosa, numpy, soundfile
- **Encoding**: FFmpeg (Vorbis codec)
- **Project Structure**: `.venv` 仮想環境（自動管理）

## ライセンス

MIT License
