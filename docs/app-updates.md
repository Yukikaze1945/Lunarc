# 软件更新

打开 **关于 Lunarc → 软件更新**。

1. 点击检查更新。要获取预发布版，请勾选“包含预发布版本”。
2. 下载新版安装程序，支持暂停和继续下载。
3. 校验完成后打开下载目录，退出 Lunarc，再运行安装程序。

更新会保留版本匹配的已安装组件。软件不会自动运行安装程序，也不会自动升级运行组件。

## 下载来源与校验

更新来自 [Lunarc Releases](https://github.com/Yukikaze1945/Lunarc-DLSS5/releases)，不需要 GitHub 账号。

下载前检查 `INSTALLER.json`、GitHub 文件信息和 `SHA256SUMS.txt`。文件下载完成后核对大小及 SHA-256，校验失败的文件不能进入安装步骤。

下载保存在应用用户数据目录的 `updates/<版本号>`。网络连接或 GitHub 限流可能导致检查失败，稍后可以重试。

## 开发说明

更新器在 Electron 主进程下载文件。未完成文件以 `.part` 保存，续传时检查 HTTP Content-Range；服务器不支持续传时重新下载。

检查超时为 20 秒，下载 60 秒没有收到数据时暂停。安装程序的格式声明和旧版附件格式兼容逻辑见 `frontend/electron/appUpdates.ts`，相关测试见 `tests/node/app_updates.test.ts`。
