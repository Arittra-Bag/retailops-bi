"""
PDF Report Generator with Markdown Support
Converts AI-generated reports to beautifully formatted PDFs
Handles bullets, headers, bold, numbering, and other markdown elements
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, blue, darkblue, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.platypus import Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
import markdown

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIReportPDFGenerator:
    """
    Advanced PDF generator for AI-generated reports
    Handles all markdown formatting elements with professional styling
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "data" / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for different elements"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=20,
            textColor=darkblue,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=15,
            textColor=blue,
            alignment=TA_LEFT
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=darkblue,
            alignment=TA_LEFT
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=10,
            textColor=black,
            alignment=TA_LEFT
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            leftIndent=20,
            bulletIndent=10,
            alignment=TA_LEFT
        ))
        
        # Numbered list style
        self.styles.add(ParagraphStyle(
            name='NumberedList',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            leftIndent=25,
            alignment=TA_LEFT
        ))
        
        # Insight box style
        self.styles.add(ParagraphStyle(
            name='InsightBox',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            spaceBefore=5,
            leftIndent=15,
            rightIndent=15,
            borderColor=blue,
            borderWidth=1,
            borderPadding=5,
            backColor=Color(0.9, 0.9, 1.0, alpha=0.3),
            alignment=TA_JUSTIFY
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='MetricStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=5,
            textColor=darkblue,
            alignment=TA_LEFT
        ))
    
    def _parse_markdown_content(self, content: str) -> List[Any]:
        """Parse markdown content and convert to ReportLab elements"""
        elements = []
        
        if not content or content.strip() == "":
            return elements
        
        # Split content into lines for processing
        lines = content.split('\n')
        current_section = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Handle headers (# ## ###)
            if line.startswith('#'):
                # First, add any accumulated content
                if current_section:
                    elements.extend(self._process_text_section('\n'.join(current_section)))
                    current_section = []
                
                # Process header
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                
                if header_level == 1:
                    elements.append(Paragraph(header_text, self.styles['CustomSubtitle']))
                elif header_level == 2:
                    elements.append(Paragraph(header_text, self.styles['SectionHeader']))
                else:
                    elements.append(Paragraph(header_text, self.styles['SubsectionHeader']))
                
                elements.append(Spacer(1, 5))
            
            # Handle bullet points (- * +)
            elif line.startswith(('- ', '* ', '+ ')):
                # Add accumulated content first
                if current_section:
                    elements.extend(self._process_text_section('\n'.join(current_section)))
                    current_section = []
                
                # Process bullet list
                bullet_items = []
                while i < len(lines) and lines[i].strip().startswith(('- ', '* ', '+ ')):
                    bullet_text = lines[i].strip()[2:].strip()
                    bullet_text = self._format_inline_markdown(bullet_text)
                    bullet_items.append(f"• {bullet_text}")
                    i += 1
                i -= 1  # Adjust for loop increment
                
                for item in bullet_items:
                    elements.append(Paragraph(item, self.styles['BulletPoint']))
                elements.append(Spacer(1, 5))
            
            # Handle numbered lists (1. 2. etc.)
            elif re.match(r'^\d+\.', line):
                # Add accumulated content first
                if current_section:
                    elements.extend(self._process_text_section('\n'.join(current_section)))
                    current_section = []
                
                # Process numbered list
                numbered_items = []
                while i < len(lines) and re.match(r'^\d+\.', lines[i].strip()):
                    match = re.match(r'^(\d+\.)(.+)', lines[i].strip())
                    if match:
                        num, text = match.groups()
                        formatted_text = self._format_inline_markdown(text.strip())
                        numbered_items.append(f"{num} {formatted_text}")
                    i += 1
                i -= 1  # Adjust for loop increment
                
                for item in numbered_items:
                    elements.append(Paragraph(item, self.styles['NumberedList']))
                elements.append(Spacer(1, 5))
            
            # Handle special AI insights (lines starting with 🤖, 📊, etc.)
            elif re.match(r'^[🤖📊💡🎯📈💰🔍⚡🚀]+', line):
                if current_section:
                    elements.extend(self._process_text_section('\n'.join(current_section)))
                    current_section = []
                
                formatted_text = self._format_inline_markdown(line)
                elements.append(Paragraph(formatted_text, self.styles['InsightBox']))
                elements.append(Spacer(1, 5))
            
            # Regular text - accumulate
            else:
                current_section.append(line)
            
            i += 1
        
        # Process any remaining content
        if current_section:
            elements.extend(self._process_text_section('\n'.join(current_section)))
        
        return elements
    
    def _process_text_section(self, text: str) -> List[Any]:
        """Process a section of regular text"""
        elements = []
        
        if not text.strip():
            return elements
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                formatted_para = self._format_inline_markdown(para.strip())
                
                # Check if it's a metric or key insight
                if any(keyword in para.lower() for keyword in ['revenue', 'customer', 'growth', 'increase', 'decrease', '%', '$', '£']):
                    elements.append(Paragraph(formatted_para, self.styles['MetricStyle']))
                else:
                    elements.append(Paragraph(formatted_para, self.styles['Normal']))
                
                elements.append(Spacer(1, 5))
        
        return elements
    
    def _format_inline_markdown(self, text: str) -> str:
        """Format inline markdown elements (bold, italic, etc.)"""
        
        # Bold text (**text** or __text__)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
        
        # Italic text (*text* or _text_)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
        
        # Code text (`text`)
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
        
        # Handle currency and percentages with special formatting
        text = re.sub(r'(£[\d,]+\.?\d*)', r'<b>\1</b>', text)
        text = re.sub(r'(\$[\d,]+\.?\d*)', r'<b>\1</b>', text)
        text = re.sub(r'([\d.]+%)', r'<b>\1</b>', text)
        
        return text
    
    def generate_pdf_from_json_report(self, report_data: Dict[str, Any]) -> str:
        """Generate PDF from JSON report data"""
        try:
            logger.info("🔄 Generating PDF report from AI insights...")
            
            # Create filename
            report_id = report_data.get('report_id', f"ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            pdf_filename = f"{report_id}.pdf"
            pdf_path = self.reports_dir / pdf_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build story (content)
            story = []
            
            # Title page
            story.append(Paragraph("🤖 AI-Powered Business Intelligence Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Report metadata
            generated_at = report_data.get('generated_at', datetime.now().isoformat())
            story.append(Paragraph(f"Generated: {generated_at}", self.styles['Normal']))
            story.append(Paragraph(f"Report ID: {report_id}", self.styles['Normal']))
            story.append(Spacer(1, 30))
            
            # Executive Summary
            story.append(Paragraph("📊 Executive Summary", self.styles['CustomSubtitle']))
            story.append(Spacer(1, 10))
            
            summary_text = """
            This comprehensive AI-generated report provides data-driven insights into retail performance, 
            customer behavior, and business opportunities. The analysis leverages advanced machine learning 
            algorithms to identify trends, segment customers, and forecast sales performance.
            """
            story.append(Paragraph(summary_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Process each section
            sections = report_data.get('sections', {})
            
            for section_name, section_data in sections.items():
                if section_data.get('status') == 'success':
                    
                    # Section title
                    section_title = section_name.replace('_', ' ').title()
                    story.append(Paragraph(f"📈 {section_title}", self.styles['CustomSubtitle']))
                    story.append(Spacer(1, 10))
                    
                    # Section insights
                    insights = section_data.get('insights', '')
                    if insights:
                        story.extend(self._parse_markdown_content(insights))
                    
                    # Section recommendations
                    recommendations = section_data.get('recommendations', '')
                    if recommendations:
                        story.append(Paragraph("💡 Recommendations", self.styles['SectionHeader']))
                        story.extend(self._parse_markdown_content(recommendations))
                    
                    story.append(Spacer(1, 20))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("---", self.styles['Normal']))
            story.append(Paragraph("Generated by RetailOps ML Analytics Platform", self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"✅ PDF report generated: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating PDF: {str(e)}")
            raise
    
    def generate_pdf_from_quick_insights(self, insights_text: str, query: str = "") -> str:
        """Generate PDF from quick insights text"""
        try:
            logger.info("🔄 Generating PDF from quick insights...")
            
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_filename = f"quick_insights_{timestamp}.pdf"
            pdf_path = self.reports_dir / pdf_filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            
            # Build story
            story = []
            
            # Title
            story.append(Paragraph("🤖 Quick AI Insights", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Query (if provided)
            if query:
                story.append(Paragraph("🔍 Query", self.styles['SectionHeader']))
                story.append(Paragraph(query, self.styles['InsightBox']))
                story.append(Spacer(1, 15))
            
            # Insights
            story.append(Paragraph("💡 AI Analysis", self.styles['SectionHeader']))
            story.extend(self._parse_markdown_content(insights_text))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("---", self.styles['Normal']))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
            story.append(Paragraph("RetailOps ML Analytics Platform", self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"✅ Quick insights PDF generated: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating quick insights PDF: {str(e)}")
            raise


def main():
    """Test the PDF generator"""
    try:
        generator = AIReportPDFGenerator()
        
        # Test with sample data
        sample_report = {
            "report_id": "test_report_20241201",
            "generated_at": datetime.now().isoformat(),
            "sections": {
                "business_summary": {
                    "status": "success",
                    "insights": """
                    ## 📊 Business Performance Overview
                    
                    My analysis reveals **strong growth patterns** across multiple metrics:
                    
                    - **Revenue Growth**: £82,793.85 total revenue
                    - **Customer Base**: 162 active customers analyzed
                    - **Product Performance**: Top-performing categories identified
                    
                    ### Key Metrics:
                    1. **Average Customer Value**: £511.68
                    2. **Customer Segments**: 6 distinct clusters identified
                    3. **Growth Trend**: Positive trajectory in Q4
                    
                    🎯 **Strategic Insight**: High-value customers (top 20%) contribute 60% of total revenue.
                    """,
                    "recommendations": """
                    💡 **Immediate Actions**:
                    - Focus retention efforts on high-value customer segments
                    - Implement targeted marketing for emerging customer groups
                    - Optimize product mix based on performance analytics
                    """
                }
            }
        }
        
        # Generate test PDF
        pdf_path = generator.generate_pdf_from_json_report(sample_report)
        print(f"✅ Test PDF generated: {pdf_path}")
        
        return pdf_path
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()