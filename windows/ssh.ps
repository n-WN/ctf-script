.\tun2proxy-bin.exe `
  --proxy  socks5://user:name@ip:port `
  --tun    utun8 `
  --setup

# 1) 读公钥
$pub = Get-Content $clientPubKeyPath -Raw

# 2) 声明服务器登录名
$remoteUser  = "yourAdminUser"   # ← 改成你的管理员帐户
$remoteHost  = "192.168.1.2"   # ← 改成 IP / 主机名 / "user@domain@host"

# 3) 生成服务器端一次性执行脚本
$remoteCmd = @"
powershell -NoLogo -NoProfile -Command "
# 新建目标目录&文件
`$targetPath = '$env:ProgramData\ssh\administrators_authorized_keys'
ni (Split-Path `$targetPath) -Force | Out-Null
Add-Content `$targetPath '$pub' -Force

# 锁权限：仅管理员 & 系统
icacls.exe `$targetPath /inheritance:r  /grant *S-1-5-32-544:F /grant *S-1-5-18:F
" 
"@

ssh $remoteUser@$remoteHost $remoteCmd
