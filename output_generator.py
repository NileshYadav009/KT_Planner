"""
Structured Output Generator
- Convert KT to Markdown, JSON, SOP format
- Auto-generate documentation
- Export for different use cases
- Coverage checklist generation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path


class StructuredOutputGenerator:
    """Generate structured outputs from KT data."""
    
    def __init__(self, job_id: str, transcript: str, kt_data: Dict):
        self.job_id = job_id
        self.transcript = transcript
        self.kt_data = kt_data
        self.timestamp = datetime.utcnow().isoformat()

    def generate_markdown_documentation(self, include_code_refs: bool = True) -> str:
        """Generate clean Markdown documentation."""
        sections_data = self.kt_data.get('section_content', {})
        schema = self.kt_data.get('schema', [])
        
        lines = []
        
        # Header
        lines.append("# Knowledge Transfer Documentation")
        lines.append(f"*Generated on: {self.timestamp}*")
        lines.append(f"*Job ID: {self.job_id}*")
        lines.append("")
        
        # Table of Contents
        lines.append("## Table of Contents")
        for section in schema:
            section_id = section.get('id')
            title = section.get('title', section_id)
            anchor = title.lower().replace(' ', '-')
            lines.append(f"- [{title}](#{anchor})")
        lines.append("")
        
        # Executive Summary
        coverage = self.kt_data.get('coverage', {})
        total_sections = len(coverage)
        covered = sum(1 for c in coverage.values() if c.get('status') in ('covered', 'weak'))
        lines.append("## Executive Summary")
        lines.append(f"- **Coverage**: {covered}/{total_sections} sections ({int(100*covered/max(total_sections,1))}%)")
        lines.append(f"- **Status**: {'✅ Complete' if covered == total_sections else '⚠️ Incomplete'}")
        lines.append("")
        
        # Main Sections
        for section in schema:
            section_id = section.get('id')
            title = section.get('title', section_id)
            required = section.get('required', False)
            
            lines.append(f"## {title}")
            if required:
                lines.append("*[REQUIRED SECTION]*")
            lines.append("")
            
            # Description
            if section.get('description'):
                lines.append(f"{section.get('description')}")
                lines.append("")
            
            # Content from KT
            sec_content = sections_data.get(section_id, {})
            sentences = sec_content.get('sentences', [])
            
            if sentences:
                lines.append("### Content")
                for sent_idx, sent in enumerate(sentences, 1):
                    text = sent.get('text', '')
                    confidence = sent.get('confidence', 1.0)
                    
                    # Add confidence indicator
                    conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                    lines.append(f"{conf_emoji} {text}")
                
                lines.append("")
                
                # Code references
                if include_code_refs:
                    code_refs = sec_content.get('code_references', [])
                    if code_refs:
                        lines.append("### Code References")
                        for ref in code_refs:
                            if isinstance(ref, dict):
                                lines.append(f"\n**File**: `{ref.get('file_name', 'unknown')}`")
                                lines.append(f"```{ref.get('language', 'text')}")
                                lines.append(ref.get('code_block', ''))
                                lines.append("```")
                        lines.append("")
            else:
                lines.append("*No content captured for this section*")
                lines.append("")
        
        # Appendix: Missing Sections
        missing = self.kt_data.get('missing_required_sections', [])
        if missing:
            lines.append("## ⚠️ Missing Required Sections")
            lines.append(f"The following {len(missing)} required sections need documentation:")
            lines.append("")
            for missing_id in missing:
                for section in schema:
                    if section.get('id') == missing_id:
                        lines.append(f"- **{section.get('title')}**: {section.get('description', 'No description')}")
            lines.append("")
        
        return "\n".join(lines)

    def generate_sop_runbook(self) -> str:
        """Generate Standard Operating Procedure runbook."""
        sections_data = self.kt_data.get('section_content', {})
        lines = []
        
        lines.append("# Standard Operating Procedure (SOP) / Runbook")
        lines.append(f"*Job ID: {self.job_id}*")
        lines.append(f"*Generated: {self.timestamp}*")
        lines.append("")
        
        # Identify procedural sections (typically involve steps, processes)
        procedural_sections = ['implementation', 'deployment', 'troubleshooting', 'maintenance']
        
        lines.append("## Quick Reference")
        lines.append("")
        
        for section_id, content in sections_data.items():
            if any(proc in section_id.lower() for proc in procedural_sections):
                title = content.get('section_title', section_id)
                lines.append(f"### {title}")
                
                sentences = content.get('sentences', [])
                for idx, sent in enumerate(sentences, 1):
                    lines.append(f"{idx}. {sent.get('text', '')}")
                
                lines.append("")
        
        lines.append("## Prerequisites")
        lines.append("- [ ] Verify system access")
        lines.append("- [ ] Check prerequisites document")
        lines.append("- [ ] Review latest policy updates")
        lines.append("")
        
        lines.append("## Execution Checklist")
        lines.append("Use this checklist to verify each step is completed:")
        lines.append("")
        
        step_num = 1
        for section_id, content in sections_data.items():
            sentences = content.get('sentences', [])
            for sent in sentences:
                lines.append(f"- [ ] Step {step_num}: {sent.get('text', '')[:80]}...")
                step_num += 1
        
        lines.append("")
        
        lines.append("## Rollback Procedure")
        rollback_section = next((s for s_id, s in sections_data.items() 
                               if 'rollback' in s_id.lower()), None)
        if rollback_section:
            lines.append(rollback_section.get('enhanced_text', 'See implementation notes'))
        else:
            lines.append("*Not documented - consult with team lead*")
        lines.append("")
        
        return "\n".join(lines)

    def generate_json_export(self) -> Dict[str, Any]:
        """Generate complete JSON export with all metadata."""
        coverage = self.kt_data.get('coverage', {})
        
        return {
            "metadata": {
                "job_id": self.job_id,
                "generated_at": self.timestamp,
                "format_version": "2.0"
            },
            "summary": {
                "total_sentences": len(self.kt_data.get('section_content', {}).get('sentences', [])),
                "coverage_percent": int(100 * sum(1 for c in coverage.values() 
                                                   if c.get('status') in ('covered', 'weak')) 
                                       / max(len(coverage), 1)),
                "missing_required_count": len(self.kt_data.get('missing_required_sections', []))
            },
            "coverage_analysis": coverage,
            "sections": self._export_sections(),
            "statistics": self.kt_data.get('statistics', {})
        }

    def _export_sections(self) -> List[Dict[str, Any]]:
        """Export sections with content."""
        sections_data = self.kt_data.get('section_content', {})
        schema = self.kt_data.get('schema', [])
        
        exported = []
        for section in schema:
            section_id = section.get('id')
            content = sections_data.get(section_id, {})
            
            exported.append({
                "id": section_id,
                "title": section.get('title'),
                "required": section.get('required', False),
                "description": section.get('description'),
                "content": {
                    "sentences": content.get('sentences', []),
                    "sentence_count": len(content.get('sentences', [])),
                    "confidence": content.get('confidence', 0.0),
                    "risk_score": content.get('risk_score', 0.0)
                }
            })
        
        return exported

    def generate_coverage_checklist(self) -> Dict[str, List[str]]:
        """Auto-generate coverage improvement suggestions."""
        missing = self.kt_data.get('missing_required_sections', [])
        coverage = self.kt_data.get('coverage', {})
        schema_map = {s.get('id'): s for s in self.kt_data.get('schema', [])}
        
        checklist = {}
        
        # For each missing section, suggest clarification questions
        for section_id in missing:
            section = schema_map.get(section_id, {})
            title = section.get('title', section_id)
            
            suggestions = self._generate_clarification_questions(section)
            checklist[title] = suggestions
        
        return checklist

    def _generate_clarification_questions(self, section: Dict) -> List[str]:
        """Generate suggested questions based on section type."""
        questions = []
        title = section.get('title', '').lower()
        description = section.get('description', '').lower()
        
        # Context-specific questions
        if 'deployment' in title or 'deploy' in description:
            questions.extend([
                "What are the deployment steps?",
                "What is the rollback procedure?",
                "How long does deployment take?",
                "What prerequisites are needed?"
            ])
        
        if 'troubleshoot' in title or 'troubleshoot' in description:
            questions.extend([
                "What are common errors?",
                "How do you diagnose issues?",
                "What are the resolution steps?",
                "When should you escalate?"
            ])
        
        if 'architecture' in title or 'overview' in title:
            questions.extend([
                "What is the system architecture?",
                "What are the key components?",
                "How do they interact?",
                "What are scalability considerations?"
            ])
        
        if 'setup' in title or 'install' in title or 'configure' in description:
            questions.extend([
                "What are the setup steps?",
                "What dependencies are needed?",
                "How do you verify the setup?",
                "What common setup issues exist?"
            ])
        
        if not questions:
            # Fallback generic questions
            questions = [
                f"What are the key aspects of {section.get('title', 'this section')}?",
                "What should team members know about this?",
                "Are there any prerequisites?",
                "What are common pitfalls?"
            ]
        
        return questions

    def generate_html_report(self, include_visuals: bool = True) -> str:
        """Generate interactive HTML report."""
        coverage = self.kt_data.get('coverage', {})
        coverage_pct = int(100 * sum(1 for c in coverage.values() 
                                    if c.get('status') in ('covered', 'weak')) 
                          / max(len(coverage), 1))
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>KT Report - {self.job_id}</title>
            <style>
                body {{ font-family: Inter, system-ui, sans-serif; margin: 40px; background: #f6f8fb; }}
                .header {{ background: linear-gradient(135deg, #0a66c2, #0b8a72); color: white; padding: 30px; border-radius: 12px; }}
                .card {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .coverage-bar {{ background: #eef3ff; height: 24px; border-radius: 999px; overflow: hidden; }}
                .coverage-fill {{ background: linear-gradient(90deg, #0a66c2, #0b8a72); height: 100%; width: {coverage_pct}%; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #0a66c2; }}
                .status-good {{ color: #10b981; }} .status-warning {{ color: #f59e0b; }} .status-error {{ color: #ef4444; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
                th {{ background: #f9fafb; font-weight: 600; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Knowledge Transfer Report</h1>
                <p>Job ID: {self.job_id}</p>
                <p>Generated: {self.timestamp}</p>
            </div>
            
            <div class="card">
                <h2>Coverage Summary</h2>
                <div class="coverage-bar">
                    <div class="coverage-fill"></div>
                </div>
                <p><strong>{coverage_pct}%</strong> of required sections documented</p>
            </div>
            
            <div class="card">
                <h2>Section Coverage</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Section</th>
                            <th>Status</th>
                            <th>Confidence</th>
                            <th>Sentences</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for section_id, cov in coverage.items():
            status = cov.get('status', 'unknown')
            status_class = 'status-good' if status == 'covered' else 'status-warning' if status == 'weak' else 'status-error'
            conf = cov.get('confidence', 0.5)
            count = cov.get('sentence_count', 0)
            
            html += f"""
                        <tr>
                            <td>{cov.get('title', section_id)}</td>
                            <td class="{status_class}">● {status.upper()}</td>
                            <td>{int(conf * 100)}%</td>
                            <td>{count}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        return html

    def save_all_formats(self, output_dir: str) -> Dict[str, str]:
        """Save in all formats to output directory."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Markdown
        md_path = f"{output_dir}/documentation.md"
        with open(md_path, 'w') as f:
            f.write(self.generate_markdown_documentation())
        paths['markdown'] = md_path
        
        # SOP/Runbook
        sop_path = f"{output_dir}/runbook.md"
        with open(sop_path, 'w') as f:
            f.write(self.generate_sop_runbook())
        paths['sop'] = sop_path
        
        # JSON
        json_path = f"{output_dir}/export.json"
        with open(json_path, 'w') as f:
            json.dump(self.generate_json_export(), f, indent=2)
        paths['json'] = json_path
        
        # HTML
        html_path = f"{output_dir}/report.html"
        with open(html_path, 'w') as f:
            f.write(self.generate_html_report())
        paths['html'] = html_path
        
        # Checklist
        checklist_path = f"{output_dir}/coverage_checklist.json"
        with open(checklist_path, 'w') as f:
            json.dump(self.generate_coverage_checklist(), f, indent=2)
        paths['checklist'] = checklist_path
        
        return paths
