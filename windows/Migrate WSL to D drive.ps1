[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 1. 创建目录（以管理员身份运行）
New-Item -ItemType Directory -Path "D:\WSL\Ubuntu" -Force

# 2. 导出Ubuntu发行版（假设原名为"Ubuntu"）
wsl --export --vhd Ubuntu "D:\WSL\Ubuntu\ext4.vhdx"

# 3. 注销旧发行版（谨慎操作！）
wsl --unregister Ubuntu

# 4. 导入到新位置
wsl --import-in-place Ubuntu "D:\WSL\Ubuntu\ext4.vhdx"

# 5. 设置默认用户（需知道用户名）
wsl -d Ubuntu -e sudo sh -c "echo -e '[user]\ndefault=您的用户名' > /etc/wsl.conf"
