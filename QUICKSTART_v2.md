# KT Planner v2.0 - Quick Start Guide

## ⚡ 5-Minute Setup

### Prerequisites
- Python 3.9+
- pip
- FFmpeg installed on system

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Whisper model (first run only)
python -c "import whisper; whisper.load_model('base')"

# 3. Start the server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access Application

Open browser to: **http://localhost:8000/enhanced.html**

---

## 🎯 Your First Upload

### Step 1: Upload Audio
1. Click **📤 New Upload** button
2. Drag audio file (MP3, WAV, M4A) or click to browse
3. Wait for transcription (30-60 seconds or less with cache)

### Step 2: Review Sentences
- Left panel shows all sentences with confidence indicators
- 🟢 Green = High confidence (>80%)
- 🟡 Yellow = Medium (60-80%)  
- 🔴 Red = Low (<60%) or Confusing

### Step 3: Drag & Assign
- Drag sentences to correct KT sections on right panel
- System learns your choices
- Repeat for confusing items (red highlighted)

### Step 4: Edit & Enhance
- Click ✏️ to edit sentence text
- Click 🔗 to link code/logs
- Click ❌ to mark as confusing for later review

### Step 5: Export
- Click **📥 Export** 
- Select formats (Markdown, SOP, JSON, Checklist)
- Click "Generate Exports"
- Data appears in response

---

## 🚀 Key Features Quick Reference

| Feature | How to Use | Shortcut |
|---------|-----------|----------|
| **Drag-Drop** | Click-drag sentence to section | Mouse motion |
| **Edit** | Click ✏️ button on sentence | Modal dialog |
| **Link Code** | Click 🔗 button, paste code | Modal dialog |
| **Mark Confusing** | Click ❌ button on sentence | Instant |
| **Create Version** | Top menu → Versions | Snapshot all |
| **Export** | Top menu → Export | Multi-format |
| **Undo/Redo** | Keyboard: Ctrl+Z / Ctrl+Y | (v2.1+) |

---

## 📊 Understanding the UI

### Left Panel - Intelligent Transcript
```
📝 Sentence 1 (🟢 0.92 confidence)
   └─ [predicted section] 92% confidence
   └─ Actions: [✏️ Edit] [🔗 Code] [❌ Confusing]

📝 Sentence 2 (🔴 0.38 confidence) ⚠️ Confusing
   └─ [unclear section] 38% confidence
   └─ Actions: [✏️ Edit] [🔗 Code] [❌ Confusing]
```

**Sidebar Stats:**
- **Mapped**: Sentences assigned to sections
- **Confusing**: Low-confidence or flagged
- **Confidence**: Average confidence %
- **Code Refs**: Linked code blocks

### Right Panel - KT Sections  
```
📚 KT SECTIONS

[Section Name] [Status: COVERED/WEAK/MISSING]
└─ X sentences • 85% confidence
└─ [Drag sentences here to assign]

[Another Section] [Status: MISSING]
└─ 0 sentences • 0% confidence
└─ [Drag sentences here to assign]
```

**Status Colors:**
- 🟢 COVERED: ≥2 sentences, high confidence
- 🟡 WEAK: 1 sentence or lower confidence
- 🔴 MISSING: No sentences

---

## 🔧 Common Tasks

### Task: Fix a Misclassified Sentence

1. Find the sentence (look for low confidence 🔴)
2. Click ✏️ Edit button
3. In modal, select correct section from dropdown
4. Click "Save Changes"
5. System learns this pattern for future sentences

### Task: Add Code Reference  

1. Click 🔗 Code button on sentence
2. Fill in form:
   - File Name: `deploy.sh`
   - Language: `bash`
   - Code: Paste your code/logs
3. Click "Link Code"
4. Code now linked to that sentence

### Task: Create Documentation Snapshot

1. Complete all mapping/edits
2. Click **📜 Versions** at top
3. Click "Create Version Snapshot"
4. Enter summary: "Completed first pass"
5. Version saved with full state

### Task: Export All Formats

1. Click **📥 Export** at top
2. Check desired formats:
   - ✓ Markdown (human-readable)
   - ✓ SOP (operational runbook)
   - ✓ JSON (data format)
   - ✓ Checklist (missing items)
3. Click "Generate Exports"
4. See results in response

### Task: Review Low-Confidence Sentences

1. Look for 🔴 red confidence indicators
2. Read the text carefully
3. Choose actions:
   - **Edit**: Click ✏️ to fix text  
   - **Clarify**: Mark as "needs clarification"
   - **Reassign**: Drag to different section
4. System learns better boundaries

---

## 📈 Understanding Confidence Scores

**Confidence = How sure the AI is about section prediction**

- **0.9-1.0 (🟢 High)**: Certain. Looks good!
- **0.7-0.9**: Pretty sure. Likely correct.
- **0.6-0.7 (🟡 Medium)**: Okay. Could use review.
- **0.4-0.6**: Unsure. Consider editing.
- **<0.4 (🔴 Low)**: Very unsure. Likely wrong.

**Importance Weight** = How significant this sentence is

- Keywords: "critical", "must", "deploy", "failure" = High
- Longer sentences tend to be more important
- Affects export ordering

---

## 🎓 Workflow Examples

### Example 1: DevOps Runbook from Recording

```
1. Upload: devops_training.mp3 (45 min)
   ↓ Transcribe (cached: 0.2s)
2. Review: 287 sentences appear
3. Assign: Drag deployment sentences to "Implementation"
4. Enhance: Link deploy.sh code to deployment steps
5. Mark: Flag 3 confusing sentences
6. Export: Generate SOP runbook + checklist
7. Share: Send SOP to operations team
```

### Example 2: Architecture Documentation

```
1. Upload: arch_discussion.m4a (120 min)
   ↓ Transcribe (parallel: 25s)
2. Review: 450 sentences appear
3. Filter: Focus on high-confidence (>0.8)
4. Organize: Sort into sections (Overview, Components, etc.)
5. Link: Attach architecture diagrams as code references
6. Export: Generate Markdown documentation
7. Publish: Share with engineering team
```

### Example 3: Troubleshooting Guide

```
1. Upload: support_calls.mp4 (90 min)
2. Review: 320 sentences, ~100 marked confusing
3. Clarify: Request clarification on 15 ambiguous items
4. Refine: Edit confusing sentences for clarity
5. Organize: Map to troubleshooting framework
6. Export: Generate runbook + checklist
7. Validate: Get SM review before publishing
```

---

## ⚠️ Troubleshooting

### Issue: Upload hangs at 30%
**Solution**: 
- Check file format (MP3, WAV supported)
- Try smaller file (<200MB)
- Check available disk space

### Issue: Low confidence scores (<50%)
**Solution**:
- Audio quality may be poor (try normalize)
- Content may be technical/specialized
- Manually assign to help system learn

### Issue: Sentences missing or duplicated
**Solution**:
- Refresh page (F5)
- Check browser console for errors
- Try re-uploading file

### Issue: Export button not working
**Solution**:
- Ensure transcription completed
- Check that sentences are mapped
- Try JSON export first

---

## 🔑 Pro Tips

### Tip 1: Learn From Corrections
Every time you drag a sentence to correct section, AI learns. After 3-5 corrections on same topic, system will suggest better placements!

### Tip 2: Use Code Links
Linking code to sentences keeps context. When reviewing later, code appears right next to relevant sentence.

### Tip 3: Version Before Major Changes
If planning significant edits, create version first. Easy to revert if needed.

### Tip 4: Export Multiple Formats
- **Markdown**: Share with writers
- **SOP**: Operations teams
- **JSON**: System integrations
- **Checklist**: Coverage tracking

### Tip 5: Mark Confusing Early
Don't waste time on low-confidence items. Mark them, move on, review later with team.

---

## 📚 More Information

- **Full Features**: See [FEATURES_v2.md](FEATURES_v2.md)
- **API Docs**: http://localhost:8000/docs
- **Data Structures**: See [FEATURES_v2.md#data-structures](FEATURES_v2.md#data-structures)
- **Performance**: See [FEATURES_v2.md#performance-metrics](FEATURES_v2.md#performance-metrics)

---

## 🆘 Need Help?

### Quick Answers
1. **How do I mark a sentence as confusing?**
   → Click red ❌ button on the sentence

2. **Can I undo a change?**
   → Yes, use keyboard Ctrl+Z (v2.1+) or create new version to revert

3. **How do I get faster transcription?**
   → Files are cached. Same file=0.2s. Use optimize settings for first run.

4. **Can I link multiple code blocks?**
   → Yes, click 🔗 multiple times on same sentence

5. **How do I export?**
   → Click 📥 Export, select formats, click "Generate Exports"

### Still Stuck?
Check browser console (F12 → Console tab) for error messages, or review [FEATURES_v2.md](FEATURES_v2.md) for detailed API reference.

---

**Version**: 2.0  
**Last Updated**: March 2026  
**Status**: Ready to Use ✅
