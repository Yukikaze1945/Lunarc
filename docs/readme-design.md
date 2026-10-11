# README 视觉资产

首页采用 Cinematic × Swiss Minimalism：深炭灰、大字、严格对齐与留白，淡金色用于强调。青绿色应用 Logo 保持原样。

| 用途 | 色值 |
| --- | --- |
| 背景 | `#191919` |
| 面板 | `#262626` |
| 强调 | `#EAD27A` |
| 正文 | `#EFEEEA` |
| 次级文字 | `#AAA9A4` |

## 资产

- `assets/lunarc-banner.png`：使用内置 imagegen 生成的品牌横幅；是视觉插画，不是软件截图。2172 × 724，3:1。
- `assets/lunarc-banner.svg`：项目内绘制的可编辑矢量备选版；1440 × 420。沿用应用原 Logo，并以网格与金色光弧组织画面。
- `assets/lunarc-logo.svg`：直接复制自应用 `frontend/public/lunarc.svg`，不改色或重画。
- `assets/lunarc-workbench.png`：2.1.1 安装版截屏，使用原创合成测试视频；不是画质或性能基准。合成素材、用户目录、运行库和原始诊断证据不随首页上传。

GitHub README 不支持自定义页面 CSS，因而用横幅、真实截图、标准 Markdown 表格和可折叠说明承载布局；没有依赖远程字体或外部徽章服务。

## 横幅生成提示词

使用内置 imagegen，未使用 CLI / API 密钥。提示词如下：

```text
Use case: ads-marketing. Asset type: GitHub README hero banner for Lunarc, a Windows video enhancement workbench. Generate a finished ultra-wide horizontal banner, around 1440 x 480 (3:1), cinematic x Swiss minimalism editorial layout. Deep charcoal flat background #191919, warm ivory #EFEEEA typography, restrained pale gold #EAD27A detail, secondary gray #AAA9A4. On the left, enormous precise bold Swiss sans-serif text 'Lunarc.' aligned to a strict grid, followed by smaller pale-gold exact text 'Light in Motion.' and below smaller ivory Chinese exact text '让视频增强的每一步，都清晰可控。'. Top left tiny text 'VIDEO ENHANCEMENT WORKBENCH', top right tiny text 'WINDOWS x64'. Bottom horizontal hairline and small labels 'ENHANCE / COMPARE / EXPORT' at left, 'SOURCE EDITION' at right. On the right, a sophisticated abstract incomplete circular light arc constructed of a few restrained gold strokes with slight cinematic glow against delicate low-opacity orthogonal grid lines, suggesting movement and reconstruction. Asymmetric, lots of empty space, meticulous typography. No fake UI, no screenshots, no download button, no NVIDIA branding, no invented logo, no registered-trademark symbol, no badges. The gold arc is abstract artwork, not a replacement brand logo. Crisp restrained premium editorial design, not cyberpunk, no purple, no flashy chrome or 3D lettering.
```

2026-10-12 更新：横幅中文改为“视频增强与插帧”，页脚改为“GPL-3.0-or-later”。首页提供安装版下载。

文字修改提示词：

```text
Change only the Chinese sentence to 视频增强与插帧 and the bottom-right SOURCE EDITION to GPL-3.0-or-later. Preserve the existing banner composition, typography, colours, gold arc and other text.
```
