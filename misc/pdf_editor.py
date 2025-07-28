#!/usr/bin/env python3
"""
Advanced PDF Editor CLI - Uses multiple PDF libraries for better compatibility
Supports: merge, extract pages, repair corrupted PDFs
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# Try to import different PDF libraries
AVAILABLE_BACKENDS = []

try:
    import fitz  # PyMuPDF
    AVAILABLE_BACKENDS.append('pymupdf')
except ImportError:
    pass

try:
    import pikepdf
    AVAILABLE_BACKENDS.append('pikepdf')
except ImportError:
    pass

try:
    from pypdf import PdfReader, PdfWriter
    AVAILABLE_BACKENDS.append('pypdf')
except ImportError:
    pass

if not AVAILABLE_BACKENDS:
    print("Error: No PDF libraries found. Install at least one of:")
    print("  pip install pymupdf    # Recommended for problematic PDFs")
    print("  pip install pikepdf    # Good alternative")
    print("  pip install pypdf      # Basic support")
    sys.exit(1)


class AdvancedPDFEditor:
    """Advanced PDF editor with multiple backend support"""
    
    def __init__(self, backend: str = 'auto', verbose: bool = False):
        self.backend = backend
        self.setup_logging(verbose)
        self.logger.info(f"Available backends: {', '.join(AVAILABLE_BACKENDS)}")
        
    def setup_logging(self, verbose: bool):
        """Configure logging"""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def merge_with_pymupdf(self, input_files: List[str], output_file: str) -> bool:
        """Merge PDFs using PyMuPDF"""
        try:
            import fitz
            
            output_pdf = fitz.open()
            successful_files = 0
            
            for pdf_file in input_files:
                try:
                    self.logger.info(f"Processing {pdf_file} with PyMuPDF...")
                    
                    # Try to open and process the PDF
                    input_pdf = fitz.open(pdf_file)
                    
                    # Insert all pages
                    output_pdf.insert_pdf(input_pdf)
                    
                    self.logger.info(f"Added {input_pdf.page_count} pages from {pdf_file}")
                    input_pdf.close()
                    successful_files += 1
                    
                except Exception as e:
                    self.logger.error(f"PyMuPDF error with {pdf_file}: {e}")
                    # Try to recover what we can
                    try:
                        input_pdf = fitz.open(pdf_file)
                        for page_num in range(input_pdf.page_count):
                            try:
                                output_pdf.insert_pdf(input_pdf, from_page=page_num, to_page=page_num)
                            except:
                                self.logger.warning(f"Skipped corrupted page {page_num + 1} from {pdf_file}")
                        input_pdf.close()
                        successful_files += 1
                    except:
                        self.logger.error(f"Could not recover any pages from {pdf_file}")
            
            if successful_files > 0:
                output_pdf.save(output_file)
                output_pdf.close()
                self.logger.info(f"Successfully merged {successful_files}/{len(input_files)} files")
                return True
            else:
                output_pdf.close()
                return False
                
        except Exception as e:
            self.logger.error(f"PyMuPDF merge failed: {e}")
            return False
    
    def merge_with_pikepdf(self, input_files: List[str], output_file: str) -> bool:
        """Merge PDFs using pikepdf"""
        try:
            import pikepdf
            
            pdf = pikepdf.new()
            successful_files = 0
            
            for pdf_file in input_files:
                try:
                    self.logger.info(f"Processing {pdf_file} with pikepdf...")
                    
                    src = pikepdf.open(pdf_file)
                    pdf.pages.extend(src.pages)
                    
                    self.logger.info(f"Added {len(src.pages)} pages from {pdf_file}")
                    successful_files += 1
                    
                except Exception as e:
                    self.logger.error(f"pikepdf error with {pdf_file}: {e}")
            
            if successful_files > 0:
                pdf.save(output_file)
                self.logger.info(f"Successfully merged {successful_files}/{len(input_files)} files")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"pikepdf merge failed: {e}")
            return False
    
    def merge_with_pypdf(self, input_files: List[str], output_file: str) -> bool:
        """Merge PDFs using pypdf"""
        try:
            from pypdf import PdfReader, PdfWriter
            
            writer = PdfWriter()
            successful_files = 0
            
            for pdf_file in input_files:
                try:
                    self.logger.info(f"Processing {pdf_file} with pypdf...")
                    
                    reader = PdfReader(pdf_file, strict=False)
                    for page in reader.pages:
                        writer.add_page(page)
                    
                    self.logger.info(f"Added {len(reader.pages)} pages from {pdf_file}")
                    successful_files += 1
                    
                except Exception as e:
                    self.logger.error(f"pypdf error with {pdf_file}: {e}")
            
            if successful_files > 0:
                with open(output_file, 'wb') as output:
                    writer.write(output)
                self.logger.info(f"Successfully merged {successful_files}/{len(input_files)} files")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"pypdf merge failed: {e}")
            return False
    
    def merge_pdfs(self, input_files: List[str], output_file: str) -> None:
        """Merge PDFs using the best available backend"""
        backends_to_try = []
        
        if self.backend == 'auto':
            # Try backends in order of robustness
            if 'pymupdf' in AVAILABLE_BACKENDS:
                backends_to_try.append(('pymupdf', self.merge_with_pymupdf))
            if 'pikepdf' in AVAILABLE_BACKENDS:
                backends_to_try.append(('pikepdf', self.merge_with_pikepdf))
            if 'pypdf' in AVAILABLE_BACKENDS:
                backends_to_try.append(('pypdf', self.merge_with_pypdf))
        else:
            # Use specific backend
            if self.backend == 'pymupdf' and 'pymupdf' in AVAILABLE_BACKENDS:
                backends_to_try.append(('pymupdf', self.merge_with_pymupdf))
            elif self.backend == 'pikepdf' and 'pikepdf' in AVAILABLE_BACKENDS:
                backends_to_try.append(('pikepdf', self.merge_with_pikepdf))
            elif self.backend == 'pypdf' and 'pypdf' in AVAILABLE_BACKENDS:
                backends_to_try.append(('pypdf', self.merge_with_pypdf))
            else:
                raise ValueError(f"Backend '{self.backend}' not available")
        
        for backend_name, merge_func in backends_to_try:
            self.logger.info(f"Trying {backend_name} backend...")
            if merge_func(input_files, output_file):
                self.logger.info(f"Successfully merged using {backend_name}")
                return
        
        raise Exception("All backends failed to merge the PDFs")
    
    def repair_pdf(self, input_file: str, output_file: str) -> None:
        """Attempt to repair a corrupted PDF"""
        repaired = False
        
        # Try PyMuPDF first (most robust)
        if 'pymupdf' in AVAILABLE_BACKENDS:
            try:
                import fitz
                self.logger.info("Attempting repair with PyMuPDF...")
                
                doc = fitz.open(input_file)
                # Save with garbage collection and compression
                doc.save(output_file, garbage=4, deflate=True, clean=True)
                doc.close()
                
                self.logger.info(f"Successfully repaired with PyMuPDF")
                repaired = True
            except Exception as e:
                self.logger.warning(f"PyMuPDF repair failed: {e}")
        
        # Try pikepdf if PyMuPDF failed
        if not repaired and 'pikepdf' in AVAILABLE_BACKENDS:
            try:
                import pikepdf
                self.logger.info("Attempting repair with pikepdf...")
                
                pdf = pikepdf.open(input_file)
                pdf.save(output_file, compress_streams=True)
                
                self.logger.info(f"Successfully repaired with pikepdf")
                repaired = True
            except Exception as e:
                self.logger.warning(f"pikepdf repair failed: {e}")
        
        if not repaired:
            raise Exception("Could not repair PDF with any available backend")
    
    def extract_text(self, input_file: str) -> None:
        """Extract text from PDF for debugging"""
        extracted = False
        
        if 'pymupdf' in AVAILABLE_BACKENDS:
            try:
                import fitz
                doc = fitz.open(input_file)
                
                print(f"\nText extraction from {input_file}:")
                print("-" * 50)
                
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        print(f"\n--- Page {page_num + 1} ---")
                        print(text[:500] + "..." if len(text) > 500 else text)
                
                doc.close()
                extracted = True
            except Exception as e:
                self.logger.error(f"Text extraction failed: {e}")
        
        if not extracted:
            print("Text extraction requires PyMuPDF")
    
    def delete_pages(self, input_file: str, output_file: str, pages: List[int]) -> None:
        """Delete specified pages from PDF"""
        deleted = False
        
        # Try PyMuPDF first
        if 'pymupdf' in AVAILABLE_BACKENDS:
            try:
                import fitz
                self.logger.info("Deleting pages with PyMuPDF...")
                
                doc = fitz.open(input_file)
                pages_to_delete = sorted(pages, reverse=True)  # Delete from end to start
                
                for page_num in pages_to_delete:
                    if 1 <= page_num <= doc.page_count:
                        doc.delete_page(page_num - 1)  # Convert to 0-indexed
                        self.logger.info(f"Deleted page {page_num}")
                    else:
                        self.logger.warning(f"Page {page_num} out of range")
                
                doc.save(output_file)
                doc.close()
                
                self.logger.info(f"Successfully deleted {len(pages)} pages")
                deleted = True
            except Exception as e:
                self.logger.warning(f"PyMuPDF delete failed: {e}")
        
        # Try pikepdf if PyMuPDF failed
        if not deleted and 'pikepdf' in AVAILABLE_BACKENDS:
            try:
                import pikepdf
                self.logger.info("Deleting pages with pikepdf...")
                
                pdf = pikepdf.open(input_file)
                pages_to_keep = []
                
                for i in range(len(pdf.pages)):
                    if (i + 1) not in pages:  # Convert to 1-indexed
                        pages_to_keep.append(i)
                
                # Create new PDF with only kept pages
                new_pdf = pikepdf.new()
                for i in pages_to_keep:
                    new_pdf.pages.append(pdf.pages[i])
                
                new_pdf.save(output_file)
                self.logger.info(f"Successfully deleted {len(pages)} pages")
                deleted = True
            except Exception as e:
                self.logger.warning(f"pikepdf delete failed: {e}")
        
        if not deleted:
            raise Exception("Could not delete pages with any available backend")


def parse_page_range(page_str: str) -> List[int]:
    """Parse page range string (e.g., '1,3-5,7' -> [1,3,4,5,7])"""
    pages = []
    for part in page_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Advanced PDF Editor CLI - Handles problematic PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available backends: {', '.join(AVAILABLE_BACKENDS)}

Examples:
  %(prog)s merge file1.pdf file2.pdf -o output.pdf
  %(prog)s merge file1.pdf file2.pdf -o output.pdf --backend pymupdf
  %(prog)s delete input.pdf -p 3 -o output.pdf
  %(prog)s repair corrupted.pdf -o fixed.pdf
  %(prog)s extract input.pdf

Install additional backends for better compatibility:
  pip install pymupdf    # Best for corrupted PDFs
  pip install pikepdf    # Good alternative
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--backend', choices=['auto', 'pymupdf', 'pikepdf', 'pypdf'],
                       default='auto', help='PDF backend to use (default: auto)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple PDFs')
    merge_parser.add_argument('files', nargs='+', help='PDF files to merge')
    merge_parser.add_argument('-o', '--output', required=True,
                            help='Output PDF file')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete pages from PDF')
    delete_parser.add_argument('input', help='Input PDF file')
    delete_parser.add_argument('-p', '--pages', required=True,
                             help='Pages to delete (e.g., 1,3-5,7)')
    delete_parser.add_argument('-o', '--output', required=True,
                             help='Output PDF file')
    
    # Repair command
    repair_parser = subparsers.add_parser('repair', help='Repair corrupted PDF')
    repair_parser.add_argument('input', help='Input PDF file')
    repair_parser.add_argument('-o', '--output', required=True,
                             help='Output PDF file')
    
    # Extract text command
    extract_parser = subparsers.add_parser('extract', help='Extract text from PDF')
    extract_parser.add_argument('input', help='Input PDF file')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    editor = AdvancedPDFEditor(backend=args.backend, verbose=args.verbose)
    
    try:
        if args.command == 'merge':
            editor.merge_pdfs(args.files, args.output)
            
        elif args.command == 'delete':
            pages = parse_page_range(args.pages)
            editor.delete_pages(args.input, args.output, pages)
            
        elif args.command == 'repair':
            editor.repair_pdf(args.input, args.output)
            
        elif args.command == 'extract':
            editor.extract_text(args.input)
            
    except Exception as e:
        logging.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
