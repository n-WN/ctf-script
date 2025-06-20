import base64
import gzip
import re
from pathlib import Path
from urllib.parse import unquote

# 建议将 scapy 的导入放在 try-except 块中，因为它可能是一个重量级依赖
# 并且有时会产生一些告警信息，可以按需处理
try:
    from scapy.all import rdpcap, TCP
except ImportError:
    print("Scapy is not installed. Please run: pip install scapy")
    exit(1)

# --- 配置常量 ---
# PCAP_FILE = Path("gzl.pcap")
PCAP_FILE = Path("~/Downloads/gzl.pcap").expanduser()
# 从代码中提取出的固定异或密钥
XOR_KEY = b'144a6b2296333602'
# 用于从TCP载荷中定位数据的标识符和正则表达式
DATA_IDENTIFIER = b'&thisiskey='
# 正则表达式用于捕获标识符后面的所有内容
DATA_REGEX = re.compile(rb'&thisiskey=(.+)')

# 正则捕获 585e09877df47113 开始, e464d81f5a49b42e 结束的中间内容

# Gzip文件的幻数（Magic Number），用于验证解压前的数据是否正确
GZIP_MAGIC_NUMBER = b'\x1f\x8b'


def decode_payload(encoded_data: bytes, key: bytes) -> bytes | None:
    """
    对单个加密载荷执行完整的解码、解密和解压流程。

    Args:
        encoded_data: 从pcap中提取的原始加密数据。
        key: 用于XOR解密的密钥。

    Returns:
        成功解压后的明文数据，如果任何步骤失败则返回 None。
    """
    try:
        # 1. URL解码 (e.g., %2B -> +)
        url_decoded_data = unquote(encoded_data.decode('ascii')).encode('ascii')
        
        # 2. Base64解码
        base64_decoded_data = base64.b64decode(url_decoded_data)
        
        # 3. 异或(XOR)解密
        # 使用列表推导式比循环append更Pythonic且高效
        decrypted_stream = bytes([
            byte ^ key[(i + 1) % len(key)]
            for i, byte in enumerate(base64_decoded_data)
        ])
        
        # 4. Gzip解压 (先验证幻数，避免不必要的错误)
        if decrypted_stream.startswith(GZIP_MAGIC_NUMBER):
            decompressed_data = gzip.decompress(decrypted_stream)
            return decompressed_data
        else:
            # print(f"Skipping data chunk: not a valid gzip stream.")
            return None
            
    except (UnicodeDecodeError, base64.binascii.Error, gzip.BadGzipFile) as e:
        # 捕获所有可能的解码/解压异常
        # print(f"An error occurred during decoding: {e}")
        return None


def extract_data_from_pcap(pcap_path: Path):
    """
    从pcap文件中读取数据包，并提取出包含特定标识符的数据。
    使用生成器(yield)可以有效降低内存占用，特别是处理大型pcap文件时。

    Args:
        pcap_path: pcap文件的路径。

    Yields:
        匹配到的加密数据块。
    """
    if not pcap_path.exists():
        print(f"Error: Pcap file not found at '{pcap_path}'")
        return

    print(f"[*] Reading packets from '{pcap_path}'...")
    packets = rdpcap(str(pcap_path))
    
    tcp_packets = (p for p in packets if p.haslayer(TCP))

    for packet in tcp_packets:
        payload = packet[TCP].payload
        # 确保payload不为空且有Raw层
        if not hasattr(payload, 'load') or not payload.load:
            continue
        
        content = payload.load
        
        # 使用更可靠的标识符进行初步筛选
        if DATA_IDENTIFIER in content:
            # 使用re.search，因为它找到第一个匹配项就停止，比findall更高效
            match = DATA_REGEX.search(content)
            if match:
                # yield使函数成为一个生成器
                yield match.group(1)


def main():
    """
    主函数，协调数据提取和解码过程。
    """
    extracted_data = extract_data_from_pcap(PCAP_FILE)
    
    print("[*] Starting decoding process...")
    found_any = False
    for i, data_chunk in enumerate(extracted_data):
        plaintext = decode_payload(data_chunk, XOR_KEY)
        if plaintext:
            found_any = True
            print(f"--- Decoded Data Chunk {i+1} ---")
            # 尝试用UTF-8解码打印，如果失败则直接打印原始bytes
            try:
                print(plaintext.decode('utf-8'))
            except UnicodeDecodeError:
                print(plaintext)
            print("-" * (26 + len(str(i+1))))
            
    if not found_any:
        print("[*] Process finished. No valid data found or decoded.")


if __name__ == "__main__":
    main()
