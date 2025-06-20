from base64 import *
from urllib.parse import unquote
import gzip
from scapy.all import *
import re

# 初始化一个列表，用于存储从网络流量中提取的信息。
# 列表中的初始值被注释掉了，可能是一个示例。
infos = [
    #* 'S85e09877dF47113k79pNmIyMjK2MPj8MDB17OhNYZZiMg==e464d81f5a49b42e',**
]

# 使用 scapy 读取名为 'gz1.pcap' 的网络数据包文件。
# 绝对路径
packets = rdpcap('')

# 从所有数据包中筛选出 TCP 包。
tcp_packets = [p for p in packets if p.haslayer(TCP)]

# 遍历每一个 TCP 数据包。
for tcp_packet in tcp_packets:
    # 获取 TCP 层的载荷（payload）。
    payload = tcp_packet[TCP].payload

    # 检查载荷是否存在，并且是否包含特定的起始标识符
    if payload and b'585e09877df47113' in payload.load:
        # 将载荷的内容从字节解码为字符串。
        content = payload.load.decode()
        
        # 使用正则表达式查找并提取从 '' 开始到 '' 结束的整个字符串。
        # re.findall 返回一个列表，我们取第一个匹配项 [0]。
        info = re.findall(r'585e09877df47113.*?e464d81f5a49b42e', content)[0]
        
        # 将提取出的完整信息追加到 infos 列表中。
        infos.append(info)

# 定义一个用于解密的密钥（字节字符串）。
key = b'144a6b2296333602'

# 遍历 infos 列表中的每一个加密字符串。
for a in infos:
    # **【新变化】**: 对字符串进行切片，移除前 16 个和后 16 个字符。
    # 这部分是之前提取到的起始和结束标识符。
    a = a[16:-16]
    
    # 对切片后的字符串进行 URL 解码。
    a = unquote(a)
    
    # 对 URL 解码后的字符串进行 Base64 解码。
    a = b64decode(a)
    
    # 初始化一个空列表，用于存放解密后的字节。
    con = []
    
    # 遍历 Base64 解码后的字节数据 a。
    for i in range(len(a)):
        # 对 a 中的每个字节与密钥 key 中的字节进行循环异或（XOR）操作。
        con.append(a[i] ^ key[(i + 1) % len(key)])
    
    # 将存放解密后字节的列表转换为字节对象。
    a = bytes(con)
    
    # **【新变化】**: 直接对数据进行 gzip 解压。
    # 注意：此版本代码移除了对 gzip 幻数（magic number）的检查，
    # 假设所有解密后的数据都是 gzip 压缩的。
    a = gzip.decompress(a)
    
    # 打印最终解密并解压后的数据。
    print(a)
