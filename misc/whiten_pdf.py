#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fitz  # PyMuPDF
import argparse
import sys
import os

def white_out_bottom_right(input_pdf_path, output_pdf_path):
    """
    将PDF每个页面的右下角区域（高度的1/6，宽度的1/2）置为白色。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_pdf_path):
        print(f"错误: 找不到输入文件 '{input_pdf_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        # 打开输入的PDF文件
        doc = fitz.open(input_pdf_path)

        # 遍历PDF的每一页
        for page in doc:
            # 获取页面的尺寸 (width, height)
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # 定义需要置为白色的矩形区域
            # x0: 页面宽度的一半
            # y0: 页面高度的 5/6 处 (从顶部计算)
            # x1: 页面总宽度
            # y1: 页面总高度
            white_rect = fitz.Rect(page_width / 2, page_height * 5 / 6, page_width, page_height)

            # 在指定区域绘制一个白色的、无边框的矩形
            # fill=[1, 1, 1] 代表RGB颜色中的白色
            page.draw_rect(white_rect, fill=[1, 1, 1], width=0)

        # 保存修改后的PDF
        doc.save(output_pdf_path)
        doc.close()
        print(f"处理成功！文件已保存至: {output_pdf_path}")

    except Exception as e:
        print(f"处理PDF时发生未知错误: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    主函数，用于解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="一个CLI工具，用于将PDF文件每页的右下角区域涂白。",
        formatter_class=argparse.RawTextHelpFormatter, # 保持帮助信息格式
        epilog="使用示例:\n"
                 "  python %(prog)s source.pdf modified.pdf"
    )
    
    parser.add_argument(
        "input_file",
        help="需要处理的输入PDF文件路径。"
    )
    
    parser.add_argument(
        "output_file",
        help="处理后保存的输出PDF文件路径。"
    )
    
    args = parser.parse_args()
    
    white_out_bottom_right(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
