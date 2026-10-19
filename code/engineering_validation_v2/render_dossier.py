"""ReportLab rendering only; scientific results are read from completed Markdown."""
from pathlib import Path
import re,html,json,hashlib
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
O=Path(__file__).resolve().parent
font=Path('C:/Users/Alban/anaconda3/Lib/site-packages/matplotlib/mpl-data/fonts/ttf')
for name,f in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf'),('DVM','DejaVuSansMono.ttf')]:pdfmetrics.registerFont(TTFont(name,str(font/f)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB',italic='DV',boldItalic='DVB')
s=getSampleStyleSheet()
s.add(ParagraphStyle(name='BodyD',fontName='DV',fontSize=9,leading=12.5,spaceAfter=7,textColor=HexColor('#263445'),splitLongWords=True))
s.add(ParagraphStyle(name='TitleD',fontName='DVB',fontSize=19,leading=24,spaceAfter=15,textColor=HexColor('#15344f'),keepWithNext=True))
s.add(ParagraphStyle(name='H1D',fontName='DVB',fontSize=11.7,leading=15.5,spaceBefore=12,spaceAfter=7,keepWithNext=True,textColor=HexColor('#15344f')))
s.add(ParagraphStyle(name='ListD',parent=s['BodyD'],leftIndent=9,firstLineIndent=-5,spaceAfter=5))
s.add(ParagraphStyle(name='CellD',fontName='DV',fontSize=7,leading=9,splitLongWords=True))
s.add(ParagraphStyle(name='HeadD',parent=s['CellD'],fontName='DVB',textColor=white))

def markup(t):
 t=html.escape(t)
 return re.sub(r'`([^`]+)`',lambda m:'<font name="DVM" size="7.7">'+m.group(1)+'</font>',t)

def footer(canvas,doc):
 canvas.setStrokeColor(HexColor('#b4c7d4'));canvas.line(42,39,A4[0]-42,39)
 canvas.setFont('DV',7.3);canvas.setFillColor(HexColor('#526777'));canvas.drawString(42,27,'Paper B | Engineering validation v2 | 18 October 2026');canvas.drawRightString(A4[0]-42,27,str(doc.page))

def make(source,target,figure_names):
 if target.exists():
  from pypdf import PdfReader
  existing=PdfReader(target);assert all(p.extract_text().strip() for p in existing.pages),'Incomplete PDF exists; preserve and diagnose'
  return
 lines=source.read_text(encoding='utf-8').splitlines();story=[];para=[];i=0
 def flush():
  if para:story.append(Paragraph(markup(' '.join(para)),s['BodyD']));para.clear()
 while i<len(lines):
  line=lines[i]
  if not line.strip():flush();i+=1;continue
  if line.startswith('| '):
   flush();table=[]
   while i<len(lines) and lines[i].startswith('| '):
    cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
    if not all(c=='---' for c in cells):table.append(cells)
    i+=1
   n=len(table[0]);width=A4[0]-84
   # Adaptive widths: compact columns, with enough space for policy names.
   weights=[1.25 if col.lower() in ('policy','map') else 1. for col in table[0]]
   if 'policy' in [c.lower() for c in table[0]]:
    for j,c in enumerate(table[0]):
     if c.lower()=='policy':weights[j]=1.8
   widths=[width*w/sum(weights) for w in weights]
   cells=[[Paragraph(markup(v),s['HeadD'] if r==0 else s['CellD']) for v in row] for r,row in enumerate(table)]
   t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#21445d')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#eef3f6')]),('LINEBELOW',(0,-1),(-1,-1),.4,HexColor('#9cb1be'))]));story.extend([t,Spacer(1,10)]);continue
  if line.startswith('# '):
   flush();title=line[2:].replace('Paper B: final engineering validation, version 2','Paper B: engineering validation v2');story.append(Paragraph(markup(title),s['TitleD'] if not story else s['H1D']))
  elif line.startswith('## '):flush();story.append(Paragraph(markup(line[3:]),s['H1D']))
  elif line.startswith('- '):flush();story.append(Paragraph(markup(line),s['ListD']))
  else:para.append(line)
  i+=1
 flush()
 if figure_names:
  story.append(PageBreak());story.append(Paragraph('Traceable engineering figures',s['TitleD']))
  for name,caption in figure_names:
   im=Image(str(O/'06_REPORTS/figures'/f'{name}.png'));w=A4[0]-84;h=w*im.imageHeight/im.imageWidth
   story.extend([Paragraph(caption,s['H1D']),Image(str(O/'06_REPORTS/figures'/f'{name}.png'),width=w,height=h),Spacer(1,14)])
 doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=52,title=lines[0][2:],author='Paper B engineering verification',pageCompression=1)
 doc.build(story,onFirstPage=footer,onLaterPages=footer)

tasks=[('06_REPORTS/FINAL_ENGINEERING_REPORT.md','06_REPORTS/TPWRS_PAPER_B_FINAL_ENGINEERING_VALIDATION.pdf',[
 ('FIGURE_A_RESIDUAL_AND_PEAK','Figure A. Residual and maximum retained-row exceedance.'),('FIGURE_B_MATERIAL_OVERLOAD_HOURS','Figure B. Full-year material violation hours.'),('FIGURE_C_HOURLY_EXCEEDANCE_DISTRIBUTION','Figure C. Hourly maximum exceedance-duration curves.'),('FIGURE_D_MATCHED_AC_OVERLOAD','Figure D. Matched selected-case AC overload, by mapping.'),('FIGURE_E_AC_THERMAL_FEASIBILITY','Figure E. Selected-case thermal pass and peak loading.')]),
 ('02_MATCHED_AC_VALIDATION/MATCHED_PREMIUM_AC_VALIDATION_REPORT.md','02_MATCHED_AC_VALIDATION/MATCHED_PREMIUM_AC_VALIDATION_REPORT.pdf',[]),
 ('03_GENERATOR_HEADROOM/GENERATOR_HEADROOM_AND_BALANCING_REPORT.md','03_GENERATOR_HEADROOM/GENERATOR_HEADROOM_AND_BALANCING_REPORT.pdf',[]),
 ('06_REPORTS/ENGINEERING_REPRODUCIBILITY_GUIDE.md','06_REPORTS/TPWRS_PAPER_B_ENGINEERING_REPRODUCIBILITY_GUIDE.pdf',[])]
from pypdf import PdfReader
manifest=[]
for source,target,figs in tasks:
 make(O/source,O/target,figs);p=O/target;reader=PdfReader(p);assert all(page.extract_text().strip() for page in reader.pages)
 manifest.append(dict(path=str(p),pages=len(reader.pages),sha256=hashlib.sha256(p.read_bytes()).hexdigest()));print(p.name,len(reader.pages),'pages',flush=True)
(O/'07_QC_AND_INTEGRITY/PDF_BUILD_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
