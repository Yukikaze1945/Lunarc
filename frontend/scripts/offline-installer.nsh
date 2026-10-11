!macro customInit
  IfFileExists "$EXEDIR\${APP_64_NAME}" payload_exists payload_missing
  payload_missing:
    MessageBox MB_OK|MB_ICONSTOP "请将安装程序与 ${APP_64_NAME} 放在同一个文件夹后重试。此安装包不在线下载运行环境。" /SD IDOK
    SetErrorLevel 2
    Quit
  payload_exists:
    ${StdUtils.HashFile} $R0 "SHA2-512" "$EXEDIR\${APP_64_NAME}"
    StrCmp $R0 "${APP_64_HASH}" payload_verified
    MessageBox MB_OK|MB_ICONSTOP "安装数据包校验失败。请重新获取完整的安装包。" /SD IDOK
    SetErrorLevel 3
    Quit
  payload_verified:
!macroend

# The stock sidecar installer caches the archive after extraction. Preserve the
# user's offline installation set instead of moving away their only data package.
!macro customInstall
  IfFileExists "$EXEDIR\${APP_64_NAME}" preserved_archive
  CopyFiles /SILENT "$LOCALAPPDATA\${APP_PACKAGE_STORE_FILE}" "$EXEDIR\${APP_64_NAME}"
  preserved_archive:
!macroend
