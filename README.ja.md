[简体中文](README.md) · [English](README.en.md) · **日本語** · [Русский](README.ru.md)

# Lunarc — DLSS 5 による動画・画像の高画質化

<p><img src="docs/assets/lunarc-logo.svg" width="32" height="32" alt="Lunarc" /> &nbsp; <strong>LUNARC</strong></p>

![Lunarc — 動画・画像の高画質化](docs/assets/lunarc-banner.png)

**Windows x64** · **GPL-3.0-or-later**

Lunarc は DLSS 5（DLSS5 / DLSSNR）のニューラルレンダリングで動画と画像を高画質化する Windows 向けソフトウェアです。DLAA、RTX VSR による拡大、NVIDIA VFG によるフレーム補間、TrueHDR による SDR から HDR への変換、専用画像拡大モデルを組み合わせて使用できます。

[ダウンロード](https://github.com/Yukikaze1945/Lunarc-DLSS5/releases/latest) / [操作マニュアル](docs/ja/user-guide.md) / [公式サイト](https://lunarc-workbench.defaddg-9590.chatgpt.site/ja/)

本ページは現在のソースを説明します。インストーラーの搭載機能はリリースノートをご確認ください。今回の言語切り替えはソース更新に含まれ、既存インストーラーには自動では追加されません。

## 01 / ワークベンチ

![動画プレビュー・処理順序・画質設定](docs/assets/lunarc-workbench.png)

スクリーンショットの素材はデモ用です。以前の表示言語を使用している場合があります。

| 処理チェーン | 比較 | 書き出しキュー |
| --- | --- | --- |
| カードを並べ替え、設定を残して各段階を無効化。 | 元画面・処理後・中央スライダー・左右表示で確認。 | サイズ・エンコード・保存先を指定し、タスクを管理。 |

上部の言語メニュー、または **ソフトウェア設定 → 表示言語** で中文・English・日本語・Русский を選べます。本機に保存され、素材・パラメーター・キューは維持されます。技術識別子と生の診断情報は原文のままです。

## 02 / 機能

| 機能 | 対応範囲 |
| --- | --- |
| DLSSNR | 1・2・3 レイヤーを個別調整。順次処理、または実験的なマルチスケールデータ整合性融合。 |
| DLAA | ランタイムのモデルプリセット。対応可否は導入済みライブラリ次第。 |
| RTX VSR | 複数の並べ替え可能な段階、品質と強度、倍率・解像度指定。モードはランタイムと GPU に依存。 |
| オプティカルフロー | NVIDIA NVOF、AMD FidelityFX Optical Flow、RAFT-Large。RAFT の解析サイズ・反復回数を指定可能。 |
| NVIDIA VFG | 固定倍率・目標 fps、シーン切り替え検出。書き出し専用。5×–8× は実験的。 |
| 色・TrueHDR | SDR 8/10-bit と実験的 HDR10。TrueHDR のコントラスト・彩度・中間グレー・ピーク輝度・AI デバンディング。外部 NVIDIA 部品が必要。 |
| 画像 | IllustrationJaNai V3・AnimeSharp V4。重みは別途導入。PNG・JPEG・WebP・AVIF・HEIC 保存はエンコーダー次第。現在の画像出力は SDR。 |
| プリセット・範囲 | 保存・適用・既定値との差分、折りたたみ、範囲書き出しと事前処理。一致する完了済みキャッシュ範囲を緑表示。 |
| ランタイム | DLSSNR の取得・導入、ハッシュ・署名確認、キャッシュ版の選択。他の NVIDIA 部品も個別要件があります。 |

ネイティブ P010 書き出しは **MKV** で **HEVC Main10** と **AV1 Main・10-bit** の SDR 10-bit / HDR10 に対応します。非対応の組み合わせは自動で置き換えず拒否します。音声・字幕・チャプター・添付・メタデータの保持は設定とコンテナー次第です。

## 03 / インストール

1. [Releases](https://github.com/Yukikaze1945/Lunarc-DLSS5/releases/latest) から Windows x64 インストーラーを取得。
2. 初回起動で **基本実行環境** と **動画処理コンポーネント** を導入。
3. **実行と診断 → DLSSNR ランタイム** で正当に使用できるライブラリを取得・導入して選択。
4. 素材を読み込み、必要な段階を有効化。[初回操作](docs/ja/user-guide.md#quickstart) を参照。

インストール版は別途 Python・Node.js・CUDA Toolkit を入れる必要はありません。**RAFT と専用画像モデル** は任意の PyTorch/CUDA 部品と重みが必要です。ネイティブ NVIDIA/AMD フローと GPU 動画デコード・エンコードには PyTorch は不要。互換バージョンの部品は更新時も保持します。

### 動作要件と制限

- Windows x64、選択機能に対応する NVIDIA GPU とドライバー。
- NVIDIA 特効ライブラリは別依存で、それぞれのライセンスが適用されます。
- 初回導入はネット必須。AI 環境は数 GB の取得になる場合があります。
- Windows の HDR が有効なら、ネイティブ動画プレビューは FP16/scRGB を使用。無効時は SDR に変換。ファイル比較は SDR 表示です。どちらも校正済み HDR 基準ではありません。
- HDR10・多尺度融合・VFG 5×–8× は実験的です。速度は環境・素材・設定に依存し、一般的なリアルタイム性能は保証しません。

## 04 / ソースから実行

```powershell
git clone https://github.com/Yukikaze1945/Lunarc-DLSS5.git
cd Lunarc-DLSS5
```

Python 3.10+、Node.js 22.18+、FFmpeg/FFprobe、Visual Studio C++ Build Tools、Windows SDK を準備。ネイティブ部品には対応 SDK と FFmpeg 開発ファイルが必要です。`DLSS_SDK_DIR` と `FFMPEG_SDK_DIR` を指定し、`native\build_all.bat` で構築後、ルートの **`run_gui.bat`** を実行します。

依存関係のコマンド・AMD フローの追加要件は [ソース構築ガイド（中国語）](docs/getting-started.md) を参照。通常の DLSS 超解像 DLL は `nvngx_dlssnr.dll` の代わりにはなりません。

## 05 / ドキュメント

- [詳細操作マニュアル](docs/ja/user-guide.md)：処理順序・モデル・色・範囲・プリセット・画像・書き出し・問題対処。
- [導入（中国語）](docs/installing.md) · [NVIDIA 設定（中国語）](docs/native-runtime.md) · [DLSSNR 管理（中国語）](docs/dlssnr-runtime.md)。
- [GPU プレビュー（中国語）](docs/native-hdr-preview.md) · [多尺度ソルバー（中国語）](docs/multiscale-solver.md) · [検証記録（中国語）](docs/public-release.md)。
- [言語実装・翻訳保守（英語）](docs/localization.md)。

## 06 / ライセンス

Lunarc は **GPL-3.0-or-later** です。[LICENSE](LICENSE)、[第三者部品の説明](THIRD_PARTY_NOTICES.md) を参照。プロジェクトのライセンスは NVIDIA の専有ライブラリの取得・再配布権を付与しません。
