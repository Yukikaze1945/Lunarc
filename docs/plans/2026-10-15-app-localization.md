# Software and GitHub localization

User request: add English, Japanese and Russian to the desktop application and GitHub documentation, preserving Chinese. Work directly without subagents. Website localization was completed separately and is already published.

## Scope

- Persist a renderer language preference and expose it in the header and software settings; preserve app state when switching.
- Localize visible React text, tooltips, placeholders, accessibility names, guides and parameter help. Keep settings values, enum/model IDs, filenames, paths and raw diagnostic/log data unchanged.
- Translate application-owned native dialog titles through the same preference. OS-owned dialog buttons retain OS behavior.
- Add translated README and user-guide documents and language links to the actual public Lunarc repository.
- Build and check state preservation, actual control values, translated search, native dialog titles, missing catalogs, and language persistence.

## Design

Use a small pure source-text translator with exact catalog keys and placeholder patterns for messages. Add explicit `ui(...)` render sinks with a TypeScript AST codemod; no DOM text replacement and no translation of API payloads. A React external store drives rerenders without remounting the workspace. Skip `pre`, `code`, and raw log subtrees. Search includes translated labels.

## Repository state

- Working app: `F:/dlss5standaloneV2`, current initial HEAD `f82e702`, origin is the older private repository. Tracked tree was clean before this request; untracked historical artifacts must stay untouched.
- `C:/Users/18392/Downloads/Lunarc-public` contains many unrelated uncommitted changes. Do not stage or publish them.
- Fresh public checkout: `F:/dlss5standaloneV2/build/github-i18n`, cloned from `git@github.com:Yukikaze1945/Lunarc.git`, initial HEAD `e661dae33314c545f983069552fcfa468c34f6b7`. Use this clean checkout for bounded public changes, retaining its source differences.
- Initial extraction: `build/app-strings.json` has 1467 keys (24246 source characters); `build/app-ui-strings.json` excludes unused legacy export explanation text and has 1244 keys (15177 characters). The extractor is `build/extract_app_strings.cjs`. Catalog generation must also collect actually used HDR blocker messages and native dialog titles.
- Do not retry consumer translation endpoints: a Google probe returned 429 and the Microsoft anonymous endpoint returned 404. No credentials were obtained or written. Continue with model-written catalogs and terminology review.
- Website translated manual source is available at `sites/lunarc/locales/{en,ja,ru}.manual.html` for GitHub guide reuse. Website remains a separate nested repository; do not change it for this task.

## Completed

- Added Chinese/English/Japanese/Russian renderer selection in the header, settings and first-run gate, plus persisted Electron preference and native dialog titles.
- Added 1,283 shared catalog keys per translated language, including parameter help and full concatenated HDR validation messages. Render enum values and API data remain unchanged; raw filenames, preset names and logs bypass translation.
- Added translated label search and a locale subscription inside the memoized export page. Checked narrow 1000-pixel native-window header layout.
- Added three README translations, language links, four 15-chapter user guides and maintenance documentation. Nine documentation files passed local-link checks.
- `npm --prefix frontend run build` passed (existing bundle-size advisory remains). Four localization contract tests passed. Browser smoke passed all four languages, state preservation, persistence, translated search, first-run gate, minimum-width layout and zero page exceptions. Tests mocked the backend and did not run GPU inference.
- Public baseline matched the working tree's initial content for every changed tracked file. Copied only 51 scoped files to the clean public checkout and pushed commit `073e6b2` to `Yukikaze1945/Lunarc` main. No unrelated dirty public-copy files, binaries or installers were published.
- Local source build is ready through root `run_gui.bat`; existing installed packages remain unchanged.
