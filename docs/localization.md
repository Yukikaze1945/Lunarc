# Interface languages

Lunarc supports Simplified Chinese (`zh-CN`), English (`en`), Japanese (`ja`) and Russian (`ru`). Use the top language selector or Preferences → Language. The first-run component screen has the same selector.

Language is a display preference. It does not enter backend settings, queue snapshots or render signatures. Changing it rerenders labels without remounting the workspace or restarting a preview session. The renderer persists the selection in `lunarc.language.v1`; Electron also stores it in `ui-language.json` within the existing application user-data directory. Native application-owned file-dialog titles use that language. System-owned dialog buttons follow Windows.

## Translation files

- `frontend/src/i18n/en.json`
- `frontend/src/i18n/ja.json`
- `frontend/src/i18n/ru.json`

Chinese source text is the key and fallback. `ui()` belongs at display boundaries: labels, option text, help, visible statuses, titles, accessibility text and confirmation prompts. Do not translate setting values, stage IDs, model filenames, codec/profile identifiers, API payloads or raw diagnostic logs. Input values are not rewritten. Unknown backend diagnostics retain their original text rather than receiving speculative translations.

Use complete sentences with numbered placeholders (`{0}`, `{1}`) for dynamic labels; targets may reorder arguments. Avoid concatenating fragments that depend on Chinese word order. Distinguish context-dependent terms, such as height versus high quality. Search includes translated labels as well as the original terminology.

Add the same key to all three catalogs and preserve all placeholder numbers. `tools/localize-ui.cjs` documents the initial display-boundary migration; new UI should use explicit `ui()` calls. Keep technical validation messages consistent with backend checks.

## Checks

```powershell
node --test tests/node/localization.test.cjs
npm --prefix frontend run build
node tests/ui/localization-smoke.cjs
```

The browser smoke check serves the built UI on an ephemeral local port and mocks the backend. It checks language switching, persistent selection, preserved media/drafts, translated search and export controls. It does not open a GPU session or validate rendering quality.

## Documentation languages

READMEs: [中文](../README.md), [English](../README.en.md), [日本語](../README.ja.md), [Русский](../README.ru.md).

User guides: [中文](user-guide.md), [English](en/user-guide.md), [日本語](ja/user-guide.md), [Русский](ru/user-guide.md).

The guides describe current source, with experimental status and runtime requirements retained. An existing installer does not gain new language controls until an updated build is installed. Detailed engineering/build records remain in their original language and are marked as such in the translated READMEs.
