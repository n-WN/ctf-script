<?php
/**
 * 哥斯拉流量 本地解密工具
 * 用于解密一个硬编码在代码中的加密字符串，并提供多种显示方式以帮助排查乱码问题。
 */

// 配置参数
$key = '421eb7f1b8e4b3cf'; // 16位解密密钥

// 硬编码的加密数据 (Base64编码后，再加密的数据)
// IMPORTANT: Replace $encrypted_data_list with your actual string.
// This string is expected to be GZIP compressed, then XOR encrypted, then Base64 encoded.

// https://buuoj.cn/challenges#[NewStarCTF%20%E5%85%AC%E5%BC%80%E8%B5%9B%E8%B5%9B%E9%81%93]%E8%BF%98%E6%98%AF%E6%B5%81%E9%87%8F%E5%88%86%E6%9E%90
// tcp.stream eq 35
// 使用数组存储多个加密字符串，便于切换和管理
$encrypted_data_list = [
    'LbptYjdmMWI4ZX+sfpKv+HlUtwFXBhmsaLV5NGMpKGVi40opG7QeTRey+0r6rrdjgH8rTma25kl2SH4sHrI0ZgKDNlH7Kxyr8CrFKf8uA9Y0WyvPfytHrPeoea54YmZsvcRnNjdmMQ==',
    'LbptYjdmMWI4ZX%2BsfpKv%2BHlUtwFXBhmsaLV5NGMpKGVi40opG7QeTRey%2B0r6rrdjgH8rTma25kl2SH4sHrI0ZgKDNlH7Kxyr8CrFKf8uA9Y0WyvPfytHrPeoea54YmZsvcRnNjdmMQ%3D%3D',
    'LbptYjdmMWI4ZketfMqs+Pt4UU45UAFSyKkfUx0RSxrD/S6FNWbN6MfnLmIzYw==',
];

// TODO: 使用 tshark 提取流量中所有 POST Request 的数据包
// 根据哥斯拉特征, 找到密码参数, 
// 解析其对应的值(使用 php sandbox exec + echo 即为加密代码),
// 在加密代码中分析出携带请求数据的参数名, 从数据包提取参数值,
// 为请求密文;
// 接着获取 Response 数据包, 删除前后的 md5 校验码, 
// 提取 Base64 编码的字符串, 为结果密文
// 使其一一对应, 即可解码输出

// 选择要解密的字符串（默认取第一个，可根据需要切换索引）
$encrypted_data_to_decrypt = $encrypted_data_list[2];

// 解密函数 (与加密函数相同，因为XOR操作是可逆的)
function decode($data, $key)
{
    // 确保 $key 至少有16个字符，以配合 ($i + 1 & 15) 的逻辑
    $key_for_xor = $key;

    $decoded_result = ''; // 用于收集解密后的字符

    for ($i = 0; $i < strlen($data); $i++) {
        // 从密钥中获取字符，使用 ($i + 1) & 15 这种索引方式（等同于 % 16）
        $c = $key_for_xor[($i + 1) & 15];

        // 对数据字节进行 XOR 运算，并将结果添加到 $decoded_result
        $decoded_result .= ($data[$i] ^ $c);
    }
    return $decoded_result; // 返回解密后的完整字符串
}

// --- 主要逻辑 ---

// 检查要解密的变量是否为空
if (empty($encrypted_data_to_decrypt)) {
    echo "错误: 本地加密数据变量为空。没有内容可以解密。\n";
    exit();
}

echo "--- 开始解密 ---\n";

// urldecode - 只有当你的加密数据确实经过 URL 编码时才需要这一步。
// 哥斯拉通常会进行 URL 编码，因此保留它通常是好的做法。
// $encrypted_data_to_decrypt = urldecode($encrypted_data_to_decrypt);
if (strpos($encrypted_data_to_decrypt, '%') !== false) {
    $encrypted_data_to_decrypt = urldecode($encrypted_data_to_decrypt);
}
echo "处理后的加密 Base64 字符串: " . $encrypted_data_to_decrypt . "\n\n";

// 1. 进行 Base64 解码
$base64_decoded_data = base64_decode($encrypted_data_to_decrypt);

if ($base64_decoded_data === false) {
    echo "错误: Base64 解码失败。请检查输入字符串是否为有效的 Base64 编码。\n";
    exit();
}

echo "Base64 解码后的原始字节串 (Hex Dump):\n";
echo "----------------------------------------\n";
echo bin2hex($base64_decoded_data) . "\n";
echo "----------------------------------------\n\n";

// 2. 进行 XOR 解密
$decrypted_payload_gzipped = decode($base64_decoded_data, $key);

// 3. 尝试 GZIP 解压
echo "尝试 GZIP 解压...\n";
$decrypted_payload = @gzdecode($decrypted_payload_gzipped); // 使用 @ 符号抑制警告，以防数据不是 GZIP 格式

if ($decrypted_payload === false) {
    echo "错误: GZIP 解压失败。解密后的数据可能不是 GZIP 格式，或者数据已损坏。\n";
    $decrypted_payload = $decrypted_payload_gzipped; // 如果解压失败，则回退显示原始解密数据
} else {
    echo "GZIP 解压成功！\n\n";
}

echo "--- 解密完成 ---\n";
echo "解密并解压后的原始内容 (尝试作为 UTF-8 文本显示):\n";
echo "----------------------------------------\n";
// 尝试检测并转换编码，如果不是 UTF-8
// 中文常见的编码有：'UTF-8', 'GBK', 'GB2312', 'BIG5', 'CP936'
$detected_encoding = mb_detect_encoding($decrypted_payload, array('UTF-8', 'GBK', 'GB2312', 'BIG5', 'CP936'), true);
if ($detected_encoding && $detected_encoding !== 'UTF-8') {
    echo "检测到原编码: " . $detected_encoding . "。尝试转换为 UTF-8...\n";
    $display_payload = mb_convert_encoding($decrypted_payload, 'UTF-8', $detected_encoding);
} else {
    $display_payload = $decrypted_payload;
}
echo $display_payload . "\n";
echo "----------------------------------------\n\n";

echo "解密后的原始内容 (原始字节流):\n";
echo "----------------------------------------\n";
echo $decrypted_payload . "\n";
echo "----------------------------------------\n";

// 额外调试信息
// echo "\n16进制数据 (用于手动分析):\n";
// echo "----------------------------------------\n";
// echo bin2hex($decrypted_payload) . "\n";
// echo "----------------------------------------\n";

?>
