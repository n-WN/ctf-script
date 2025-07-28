#!/usr/bin/env python3
"""
PDF水印添加工具
支持给指定PDF的指定页码添加水印（支持中文和密度控制）
"""

import argparse
import os
import sys
import platform
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import Color
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
except ImportError:
    print("请先安装必要的依赖库：")
    print("pip install pypdf reportlab")
    sys.exit(1)


def find_chinese_font() -> Optional[str]:
    """
    自动查找系统中的中文字体
    
    Returns:
        字体文件路径，如果找不到返回None
    """
    system = platform.system()
    
    # 常见中文字体路径
    font_paths = []
    
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttf",    # 微软雅黑
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Songti.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
    
    # 查找第一个存在的字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    
    # 尝试查找其他可能的中文字体
    possible_dirs = []
    if system == "Windows":
        possible_dirs = ["C:/Windows/Fonts"]
    elif system == "Darwin":
        possible_dirs = ["/System/Library/Fonts", "/Library/Fonts", "~/Library/Fonts"]
    else:
        possible_dirs = ["/usr/share/fonts", "~/.fonts"]
    
    # 搜索包含CJK、Chinese、CN的字体
    for dir_path in possible_dirs:
        dir_path = os.path.expanduser(dir_path)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(('.ttf', '.ttc', '.otf')):
                    file_lower = file.lower()
                    if any(keyword in file_lower for keyword in ['cjk', 'chinese', 'cn', 'noto']):
                        return os.path.join(dir_path, file)
    
    return None


def setup_font(font_path: Optional[str] = None) -> Tuple[str, bool]:
    """
    设置字体，返回字体名称和是否支持中文
    
    Args:
        font_path: 字体文件路径，None则自动查找
        
    Returns:
        (字体名称, 是否支持中文)
    """
    if font_path:
        # 使用用户指定的字体
        try:
            font_name = "CustomFont"
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            return font_name, True
        except Exception as e:
            print(f"警告: 无法加载字体 {font_path}: {e}")
            print("将尝试使用系统字体...")
    
    # 尝试自动查找中文字体
    chinese_font = find_chinese_font()
    if chinese_font:
        try:
            font_name = "ChineseFont"
            pdfmetrics.registerFont(TTFont(font_name, chinese_font))
            print(f"使用中文字体: {os.path.basename(chinese_font)}")
            return font_name, True
        except Exception as e:
            print(f"警告: 无法加载中文字体: {e}")
    
    # 如果找不到中文字体，尝试使用内置的CID字体
    try:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        return 'STSong-Light', True
    except:
        pass
    
    # 最后的选择：使用默认字体
    print("警告: 未找到中文字体，中文可能显示为方块")
    print("建议安装中文字体或使用 --font 参数指定字体文件")
    return "Helvetica", False


def get_density_params(density: str, page_width: float, page_height: float, 
                      text_length: int) -> Tuple[int, int, float, float]:
    """
    根据密度级别获取水印参数
    
    Args:
        density: 密度级别 (low, medium, high, auto)
        page_width: 页面宽度
        page_height: 页面高度
        text_length: 文本长度
        
    Returns:
        (列数, 行数, 水平间距, 垂直间距)
    """
    if density == "auto":
        # 根据文本长度自动调整
        if text_length <= 5:
            density = "high"
        elif text_length <= 15:
            density = "medium"
        elif text_length <= 30:
            # 长文本使用适中的布局
            cols = max(2, int(page_width / 250))
            rows = max(2, int(page_height / 180))
            h_spacing = page_width / (cols + 1)
            v_spacing = page_height / (rows + 1)
            return cols, rows, h_spacing, v_spacing
        else:
            density = "low"
    
    if density == "low":
        # 稀疏布局
        cols = max(1, int(page_width / 300))
        rows = max(1, int(page_height / 200))
    elif density == "medium":
        # 中等密度
        cols = max(2, int(page_width / 200))
        rows = max(2, int(page_height / 150))
    else:  # high
        # 密集布局
        cols = max(3, int(page_width / 150))
        rows = max(3, int(page_height / 100))
    
    # 计算间距
    h_spacing = page_width / (cols + 1)
    v_spacing = page_height / (rows + 1)
    
    return cols, rows, h_spacing, v_spacing


def create_watermark(watermark_text: str, output_path: str, 
                    font_size: int = 50, opacity: float = 0.3,
                    angle: int = 45, color: tuple = (0.5, 0.5, 0.5),
                    font_path: Optional[str] = None,
                    density: str = "auto",
                    spacing: Optional[float] = None):
    """
    创建水印PDF文件
    
    Args:
        watermark_text: 水印文字
        output_path: 输出路径
        font_size: 字体大小
        opacity: 透明度 (0-1)
        angle: 旋转角度
        color: RGB颜色值 (0-1范围)
        font_path: 字体文件路径
        density: 密度级别 (low, medium, high, auto)
        spacing: 自定义间距（覆盖密度设置）
    """
    import random
    
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # 设置字体
    font_name, supports_chinese = setup_font(font_path)
    
    # 检查是否包含中文字符
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in watermark_text)
    if has_chinese and not supports_chinese:
        print("\n警告: 检测到中文字符但未找到合适的中文字体")
        print("解决方案:")
        print("1. Windows: 使用 --font C:/Windows/Fonts/simhei.ttf")
        print("2. macOS: 使用 --font '/System/Library/Fonts/STHeiti Light.ttc'")
        print("3. Linux: 安装中文字体包 (如 fonts-wqy-zenhei)")
    
    # 如果是单个水印模式（density="single"），只在中心绘制一个
    if density == "single":
        c.saveState()
        c.setFillColor(Color(*color, alpha=opacity))
        c.setFont(font_name, font_size)
        c.translate(width/2, height/2)
        c.rotate(angle)
        text_width = c.stringWidth(watermark_text, font_name, font_size)
        c.drawString(-text_width/2, 0, watermark_text)
        c.restoreState()
        c.save()
        return
    
    # 使用艺术布局（主水印+背景平铺）
    if density == "auto" and len(watermark_text) < 20:
        # 主水印 - 页面中心
        c.saveState()
        c.setFillColor(Color(*color, alpha=opacity))
        c.setFont(font_name, font_size)
        c.translate(width/2, height/2)
        c.rotate(angle)
        text_width = c.stringWidth(watermark_text, font_name, font_size)
        c.drawString(-text_width/2, 0, watermark_text)
        c.restoreState()
        
        # 背景平铺 - 使用规划好的位置避免重叠
        background_size = font_size // 2
        c.setFont(font_name, background_size)
        
        # 计算安全区域（考虑旋转后的文本边界）
        angle_rad = math.radians(angle)
        text_height = background_size
        diagonal = math.sqrt((len(watermark_text) * background_size * 0.6)**2 + text_height**2)
        safe_margin = diagonal / 2 + 30  # 额外30像素安全边距
        
        # 规划背景水印位置（3x3网格，避开中心）
        positions = [
            (width * 0.2, height * 0.8),   # 左上
            (width * 0.5, height * 0.85),  # 中上
            (width * 0.8, height * 0.8),   # 右上
            (width * 0.15, height * 0.5),  # 左中
            (width * 0.85, height * 0.5),  # 右中
            (width * 0.2, height * 0.2),   # 左下
            (width * 0.5, height * 0.15),  # 中下
            (width * 0.8, height * 0.2),   # 右下
        ]
        
        # 随机选择6-8个位置
        num_watermarks = random.randint(6, 8)
        selected_positions = random.sample(positions, num_watermarks)
        
        for x, y in selected_positions:
            # 添加小幅随机偏移
            x += random.randint(-20, 20)
            y += random.randint(-20, 20)
            rotation = angle + random.randint(-15, 15)
            size_factor = random.uniform(0.8, 1.2)
            opacity_factor = random.uniform(0.4, 0.7)
            
            c.saveState()
            c.translate(x, y)
            c.rotate(rotation)
            c.setFont(font_name, int(background_size * size_factor))
            c.setFillColor(Color(*color, alpha=opacity/2 * opacity_factor))
            
            # 居中绘制
            current_text_width = c.stringWidth(watermark_text, font_name, int(background_size * size_factor))
            c.drawString(-current_text_width/2, 0, watermark_text)
            c.restoreState()
        
        c.save()
        return
    
    # 网格布局（用于长文本或指定密度）
    # 获取密度参数
    cols, rows, h_spacing, v_spacing = get_density_params(
        density, width, height, len(watermark_text)
    )
    
    # 如果用户指定了间距，使用用户的值
    if spacing is not None:
        h_spacing = spacing
        v_spacing = spacing
        cols = max(1, int(width / h_spacing))
        rows = max(1, int(height / v_spacing))
    
    # 设置基础样式
    c.setFillColor(Color(*color, alpha=opacity))
    c.setFont(font_name, font_size)
    
    # 计算文本边界（考虑旋转）
    try:
        text_width = c.stringWidth(watermark_text, font_name, font_size)
    except:
        text_width = len(watermark_text) * font_size * 0.5
    
    text_height = font_size
    angle_rad = math.radians(angle)
    # 旋转后的边界框
    rotated_width = abs(text_width * math.cos(angle_rad)) + abs(text_height * math.sin(angle_rad))
    rotated_height = abs(text_width * math.sin(angle_rad)) + abs(text_height * math.cos(angle_rad))
    
    # 确保间距足够避免重叠
    min_h_spacing = rotated_width * 1.2  # 留出20%安全边距
    min_v_spacing = rotated_height * 1.2
    
    # 调整字体大小以适应间距
    if h_spacing < min_h_spacing or v_spacing < min_v_spacing:
        scale_factor = min(h_spacing / min_h_spacing, v_spacing / min_v_spacing)
        adjusted_font_size = int(font_size * scale_factor * 0.8)  # 额外缩小20%确保安全
        adjusted_font_size = max(10, adjusted_font_size)  # 最小10号字
    else:
        adjusted_font_size = font_size
    
    # 重新计算调整后的文本边界
    c.setFont(font_name, adjusted_font_size)
    text_width = c.stringWidth(watermark_text, font_name, adjusted_font_size)
    
    # 绘制水印网格
    for row in range(rows):
        for col in range(cols):
            x = (col + 1) * h_spacing
            y = (row + 1) * v_spacing
            
            # 添加受控的随机变化
            if density in ["medium", "high"]:
                # 限制随机偏移，确保不会造成重叠
                max_offset = min(h_spacing, v_spacing) * 0.1  # 最多10%的偏移
                x_offset = random.uniform(-max_offset, max_offset)
                y_offset = random.uniform(-max_offset, max_offset)
                x += x_offset
                y += y_offset
                
                # 其他随机变化
                rotation = angle + random.randint(-5, 5)  # 减小角度变化
                size_variation = random.uniform(0.9, 1.1)  # 减小大小变化
                opacity_variation = random.uniform(0.8, 1.0)
            else:
                rotation = angle
                size_variation = 1.0
                opacity_variation = 1.0
            
            c.saveState()
            c.translate(x, y)
            c.rotate(rotation)
            c.setFont(font_name, int(adjusted_font_size * size_variation))
            c.setFillColor(Color(*color, alpha=opacity * opacity_variation))
            
            # 重新计算文本宽度并居中绘制
            current_text_width = c.stringWidth(watermark_text, font_name, 
                                             int(adjusted_font_size * size_variation))
            c.drawString(-current_text_width/2, 0, watermark_text)
            c.restoreState()
    
    c.save()


def add_watermark_to_pdf(input_pdf: str, output_pdf: str, watermark_pdf: str, 
                        pages: Optional[List[int]] = None):
    """
    给PDF添加水印
    
    Args:
        input_pdf: 输入PDF路径
        output_pdf: 输出PDF路径
        watermark_pdf: 水印PDF路径
        pages: 要添加水印的页码列表（从1开始），None表示所有页
    """
    # 读取PDF
    pdf_reader = PdfReader(input_pdf)
    pdf_writer = PdfWriter()
    
    # 读取水印
    watermark_reader = PdfReader(watermark_pdf)
    watermark_page = watermark_reader.pages[0]
    
    # 获取总页数
    total_pages = len(pdf_reader.pages)
    
    # 处理页码
    if pages is None:
        pages = list(range(1, total_pages + 1))
    else:
        # 验证页码
        invalid_pages = [p for p in pages if p < 1 or p > total_pages]
        if invalid_pages:
            raise ValueError(f"无效的页码: {invalid_pages}. PDF共有{total_pages}页")
    
    # 逐页处理
    for i in range(total_pages):
        page_num = i + 1
        page = pdf_reader.pages[i]
        
        if page_num in pages:
            # 添加水印
            page.merge_page(watermark_page)
        
        pdf_writer.add_page(page)
    
    # 写入输出文件
    with open(output_pdf, 'wb') as output_file:
        pdf_writer.write(output_file)


def parse_pages(pages_str: str) -> List[int]:
    """
    解析页码字符串
    支持格式: "1,3,5-8,10"
    
    Args:
        pages_str: 页码字符串
        
    Returns:
        页码列表
    """
    pages = []
    parts = pages_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 处理范围
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            pages.extend(range(start, end + 1))
        else:
            # 单个页码
            pages.append(int(part))
    
    # 去重并排序
    return sorted(list(set(pages)))


def main():
    parser = argparse.ArgumentParser(
        description="PDF水印添加工具 - 支持中文和密度控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.pdf -t "机密文件" -p 1,3,5-8
  %(prog)s input.pdf -t "CONFIDENTIAL" --density high
  %(prog)s input.pdf -t "版权所有" --font-size 60 --opacity 0.2
  %(prog)s input.pdf -t "这是一个很长的水印文字示例" --density medium
  
中文支持:
  Windows: %(prog)s input.pdf -t "机密" --font C:/Windows/Fonts/simhei.ttf
  macOS:   %(prog)s input.pdf -t "机密" --font "/System/Library/Fonts/STHeiti Light.ttc"
  Linux:   %(prog)s input.pdf -t "机密" --font /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
  
密度控制:
  %(prog)s input.pdf -t "样本" --density low      # 稀疏布局
  %(prog)s input.pdf -t "样本" --density medium   # 中等密度
  %(prog)s input.pdf -t "样本" --density high     # 密集布局
  %(prog)s input.pdf -t "样本" --density single   # 单个居中水印
  %(prog)s input.pdf -t "样本" --spacing 100      # 自定义间距
  
注: 默认(auto)模式对短文本使用主水印+随机背景的艺术布局
        """
    )
    
    # 必需参数
    parser.add_argument('input', help='输入PDF文件路径')
    
    # 可选参数
    parser.add_argument('-t', '--text', default='WATERMARK', 
                       help='水印文字 (默认: WATERMARK)')
    parser.add_argument('-o', '--output', 
                       help='输出PDF文件路径 (默认: 输入文件名_watermarked.pdf)')
    parser.add_argument('-p', '--pages', default='all',
                       help='要添加水印的页码，如 "1,3,5-8" 或 "all" (默认: all)')
    
    # 水印样式参数
    parser.add_argument('--font', help='字体文件路径（用于支持中文）')
    parser.add_argument('--font-size', type=int, default=50,
                       help='字体大小 (默认: 50)')
    parser.add_argument('--opacity', type=float, default=0.3,
                       help='透明度 0-1 (默认: 0.3)')
    parser.add_argument('--angle', type=int, default=45,
                       help='旋转角度 (默认: 45)')
    parser.add_argument('--color', default='0.5,0.5,0.5',
                       help='RGB颜色值，逗号分隔，范围0-1 (默认: 0.5,0.5,0.5)')
    
    # 密度控制参数
    parser.add_argument('--density', choices=['low', 'medium', 'high', 'single', 'auto'],
                       default='auto',
                       help='水印密度: low(稀疏), medium(中等), high(密集), single(单个), auto(自动，短文本使用艺术布局) (默认: auto)')
    parser.add_argument('--spacing', type=float,
                       help='自定义水印间距（像素），会覆盖density设置')
    
    # 其他选项
    parser.add_argument('--list-fonts', action='store_true',
                       help='列出系统中找到的中文字体')
    
    args = parser.parse_args()
    
    # 如果是列出字体
    if args.list_fonts:
        print("正在查找系统中的中文字体...")
        font = find_chinese_font()
        if font:
            print(f"找到中文字体: {font}")
        else:
            print("未找到中文字体")
        
        # 列出常见字体位置
        system = platform.system()
        print(f"\n{system} 系统常见中文字体位置:")
        if system == "Windows":
            print("  C:/Windows/Fonts/simhei.ttf (黑体)")
            print("  C:/Windows/Fonts/simsun.ttc (宋体)")
            print("  C:/Windows/Fonts/msyh.ttf (微软雅黑)")
        elif system == "Darwin":
            print("  /System/Library/Fonts/STHeiti Light.ttc")
            print("  /System/Library/Fonts/PingFang.ttc")
            print("  /Library/Fonts/Songti.ttc")
        else:
            print("  /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
            print("  /usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")
        
        sys.exit(0)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 '{args.input}'")
        sys.exit(1)
    
    # 设置输出文件名
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_watermarked.pdf")
    
    # 解析页码
    if args.pages.lower() == 'all':
        pages = None
    else:
        try:
            pages = parse_pages(args.pages)
        except ValueError as e:
            print(f"错误: 无效的页码格式 '{args.pages}'")
            print("正确格式示例: 1,3,5-8,10")
            sys.exit(1)
    
    # 解析颜色
    try:
        color = tuple(float(x) for x in args.color.split(','))
        if len(color) != 3 or any(c < 0 or c > 1 for c in color):
            raise ValueError
    except:
        print(f"错误: 无效的颜色值 '{args.color}'")
        print("颜色值应为3个0-1之间的数字，用逗号分隔")
        sys.exit(1)
    
    # 创建临时水印文件
    watermark_pdf = "temp_watermark.pdf"
    
    try:
        print(f"正在处理 '{args.input}'...")
        print(f"水印文字: '{args.text}'")
        print(f"密度设置: {args.density}")
        
        # 创建水印
        create_watermark(
            watermark_text=args.text,
            output_path=watermark_pdf,
            font_size=args.font_size,
            opacity=args.opacity,
            angle=args.angle,
            color=color,
            font_path=args.font,
            density=args.density,
            spacing=args.spacing
        )
        
        # 添加水印到PDF
        add_watermark_to_pdf(
            input_pdf=args.input,
            output_pdf=args.output,
            watermark_pdf=watermark_pdf,
            pages=pages
        )
        
        print(f"成功! 输出文件: '{args.output}'")
        
        if pages:
            print(f"已在以下页码添加水印: {pages}")
        else:
            print("已在所有页面添加水印")
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 清理临时文件
        if os.path.exists(watermark_pdf):
            os.remove(watermark_pdf)


if __name__ == "__main__":
    main()
