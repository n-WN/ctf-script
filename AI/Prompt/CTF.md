解决当前目录下的 CTF Chall, 每题目均新开一个主目录，新目录层级如下:

```
(env) ~/D/b/2025-UTCTF-basic-crypto> tree
 .
├── 󰂺 README.md
├──  solution
│   └──  solution.py
└──  task
    └──  attachment.txt
```

，如果解决, 写一份 markdown write-up 命名为 README.md, 要求 latex 友好（使用美元符号而不是 `\[`）、简体中文, 尽可能丰富的 [推理、数学推导] 并记录尝试过程, 直接写入文件, 并写一份 solution.py

如果有的python 包不存在，可以用 uv init 新建环境

你应该使用 sage  来运行 , 或者 uv init  创建 python 环境, 而不是直接操作系统的 python
