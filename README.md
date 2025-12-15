# Stormworks Custom Engine Sound Generator

Stormworks の Component MOD 用に、カスタムエンジン音源を含む `.bin` ファイルを自動生成する Python GUI ツールです。

## 機能

1. **動画・音声の読み込み**  
   エンジンの回転域に応じた4種類のエンジン音源（L1～L4）を読み込みます。MP4, WAV, OGG など対応形式ならOK。

2. **ゼロクロスポイント自動検出**  
   指定された秒数範囲から最も近いゼロクロス点で音声を自動切り取り。ノイズなしのシームレスループが生成されます。

3. **OGG Vorbis に直接変換**  
   中間ファイル不要。PCM バッファから直接 OGG へエンコード（品質 0～10、デフォルト 3）。

4. **XML テンプレートの自動生成**  
   エンジン名を反映した `modular_engine_crankshaft.xml` を作成します。

5. **component_mod_compiler で `.bin` 生成**  
   最終的に Stormworks が読める Component MOD バイナリを生成します。

## 必要な環境

- **Python 3.11 以上**
- **FFmpeg** がインストール済み（PATH に登録されていること）
- **PySide6, librosa, soundfile, audioread** 等（自動インストール）

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
   任意の名前を指定（例：「LFA Engine」）

### 2. ソースファイルの選択
   **「Multi-select sources…」** ボタンで 1～4 個のファイルを一括選択、または個別の「Browse…」で指定

   - 複数選択した場合、先頭から L1 → L2 → L3 → L4 に自動割り当て
   - 個別編集も可能

### 3. ループ秒数を設定
   各ステージ（L1～L4）ごとに：
   - **Start (s)** → ループ区間の開始秒（デフォルト 1.0 s）
   - **End (s)** → ループ区間の終了秒（デフォルト 3.0 s）

### 4. パラメータ調整（オプション）
   - **Zero-cross search radius** → ゼロクロス探索範囲（サンプル数、デフォルト 4000）
   - **Vorbis quality** → OGG 圧縮品質（0～10、デフォルト 3、高いほど高品質・ファイル大）

### 5. 「Generate」をクリック
   自動処理開始。ログパネルに進捗が表示されます。

## 出力ファイル

実行完了後、`results/` 直下に以下のフォルダが作成されます：

```
results/
└── <engine-name>/
    ├── <engine-name>.bin              ← Stormworks Component MOD
    ├── <engine-name>_modular_engine_crankshaft.xml
    ├── <engine-name>-L1.ogg
    ├── <engine-name>-L2.ogg
    ├── <engine-name>-L3.ogg
    └── <engine-name>-L4.ogg
```

- `.bin` ファイルを Stormworks MOD フォルダへコピーして使用
- OGG/XML は参考用に保存（再処理時に参照可能）

## トラブルシューティング

| 状況 | 対応 |
|------|------|
| GUI が起動しない | FFmpeg が PATH に無いか確認。`ffmpeg -version` で確認してください |
| コンパイル失敗 | ログで「Compiler finished」以降の出力を確認。XML/OGG に異常がないか見直してください |
| `.bin` が生成されない | `results/` フォルダが空か確認。コンパイラ側のエラーメッセージをログで確認 |

## 開発環境

- **Language**: Python 3.11+
- **GUI**: PySide6
- **Audio Processing**: librosa, soundfile
- **Encoding**: FFmpeg (Vorbis codec)
- **Project Structure**: `.venv` 仮想環境（自動管理）
