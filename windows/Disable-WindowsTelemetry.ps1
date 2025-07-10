[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 检查是否以管理员身份运行
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "请以管理员身份运行此脚本！" -ForegroundColor Red
    exit 1
}
# 1. 检查 CompatTelRunner.exe 是否在 System32 目录
$compatTelPath = "C:\Windows\System32\CompatTelRunner.exe"
if (Test-Path $compatTelPath) {
    Write-Host "✅ CompatTelRunner.exe 位于默认路径：$compatTelPath" -ForegroundColor Green
} else {
    Write-Host "⚠️ 警告：CompatTelRunner.exe 不在默认路径！可能是恶意软件！" -ForegroundColor Yellow
    Write-Host "建议使用 Process Explorer 或杀毒软件检查！" -ForegroundColor Red
}
# 2. 运行 SFC 扫描（可选）
Write-Host "`n🔍 正在运行 SFC /SCANNOW 检查系统文件..." -ForegroundColor Cyan
sfc /scannow
Write-Host "SFC 扫描完成。" -ForegroundColor Green
# 3. 禁用遥测（三种方法选其一）
Write-Host "`n🛑 正在禁用 Windows 遥测..." -ForegroundColor Cyan
### 方法 1：通过组策略（仅限专业版/企业版）
try {
    if (Get-Command "gpedit.msc" -ErrorAction SilentlyContinue) {
        Write-Host "🔄 正在通过组策略禁用遥测..." -ForegroundColor Cyan
        $gpoPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection"
        if (-not (Test-Path $gpoPath)) {
            New-Item -Path $gpoPath -Force | Out-Null
        }
        Set-ItemProperty -Path $gpoPath -Name "AllowTelemetry" -Value 0 -Type DWord
        Write-Host "✅ 组策略已禁用遥测 (AllowTelemetry=0)" -ForegroundColor Green
    } else {
        Write-Host "⚠️ 组策略编辑器 (gpedit.msc) 不可用（家庭版用户请用注册表方法）" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ 组策略配置失败: $_" -ForegroundColor Red
}
### 方法 2：禁用任务计划
try {
    Write-Host "🔄 正在禁用任务计划中的 'Microsoft Compatibility Appraiser'..." -ForegroundColor Cyan
    Disable-ScheduledTask -TaskName "Microsoft Compatibility Appraiser" -TaskPath "\Microsoft\Windows\Application Experience\" -ErrorAction Stop
    Write-Host "✅ 任务计划已禁用" -ForegroundColor Green
} catch {
    Write-Host "❌ 禁用任务计划失败: $_" -ForegroundColor Red
}
### 方法 3：直接修改注册表（适用于所有 Windows 版本）
try {
    Write-Host "🔄 正在通过注册表禁用遥测..." -ForegroundColor Cyan
    $regPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection"
    if (-not (Test-Path $regPath)) {
        New-Item -Path $regPath -Force | Out-Null
    }
    Set-ItemProperty -Path $regPath -Name "AllowTelemetry" -Value 0 -Type DWord
    Write-Host "✅ 注册表已禁用遥测 (AllowTelemetry=0)" -ForegroundColor Green
} catch {
    Write-Host "❌ 注册表修改失败: $_" -ForegroundColor Red
}
# 4. 提示重启
Write-Host "`n🔄 需要重启以使更改生效！" -ForegroundColor Yellow
$choice = Read-Host "是否现在重启？ (Y/N)"
if ($choice -eq "Y" -or $choice -eq "y") {
    Restart-Computer -Force
} else {
    Write-Host "请稍后手动重启。" -ForegroundColor Yellow
}
