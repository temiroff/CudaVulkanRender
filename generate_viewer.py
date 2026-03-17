#!/usr/bin/env python3
"""
Asset Library Viewer Generator
Scans asset_library/ and produces a single self-contained asset_viewer.html
with all metadata and render previews embedded. Re-run whenever new assets arrive.

Usage:
    python generate_viewer.py [asset_library_path]
"""

import sys, os, json, base64, pathlib, datetime, webbrowser

ASSET_DIR  = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("asset_library")
OUTPUT     = pathlib.Path("asset_viewer.html")

# ─────────────────────────────────────────────────────────────────────────────

def scan_assets(root: pathlib.Path) -> list[dict]:
    assets = []
    if not root.exists():
        print(f"[warn] Directory not found: {root}")
        return assets

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir(): continue
        for obj_dir in sorted(cat_dir.iterdir()):
            if not obj_dir.is_dir(): continue

            meta = {}
            meta_file = obj_dir / "metadata.json"
            if meta_file.exists():
                try: meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except: pass

            def load_img(fname):
                f = obj_dir / fname
                return base64.b64encode(f.read_bytes()).decode() if f.exists() else None

            assets.append({
                "category":    cat_dir.name,
                "name":        meta.get("name",        obj_dir.name),
                "object":      meta.get("object",      obj_dir.name),
                "description": meta.get("description", ""),
                "tags":        meta.get("tags",        ""),
                "date":        meta.get("date",        ""),
                "frames":      meta.get("frames_accumulated", ""),
                "res":         "×".join(str(x) for x in meta.get("render_resolution", [])),
                "nim":         meta.get("nim_used", False),
                "img_b64":       load_img("render.jpg"),
                "img_front_b64": load_img("render_front.jpg"),
                "img_side_b64":  load_img("render_side.jpg"),
            })
    return assets

# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Asset Library — {count} assets</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0e0e10;color:#c8c8c8;font-family:'Segoe UI',system-ui,sans-serif;
        font-size:14px;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
  header{{background:#161618;border-bottom:1px solid #2a2a2e;padding:11px 18px;
          display:flex;align-items:center;gap:14px;flex-shrink:0}}
  header h1{{font-size:16px;font-weight:600;color:#f0f0f0;letter-spacing:.3px;margin-right:6px}}
  header h1 span{{color:#6db3f2}}
  #search{{flex:1;max-width:320px;background:#1e1e22;border:1px solid #3a3a3e;border-radius:5px;
           color:#d8d8d8;padding:6px 11px;font-size:13px;outline:none}}
  #search:focus{{border-color:#6db3f2}}
  #search::placeholder{{color:#666}}
  #count-label{{margin-left:auto;color:#888;font-size:12px}}
  #gen-time{{color:#555;font-size:12px}}
  .body{{display:flex;flex:1;overflow:hidden}}
  aside{{width:170px;flex-shrink:0;background:#131315;border-right:1px solid #222;
         overflow-y:auto;padding:10px 0}}
  aside h2{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
            color:#666;padding:0 12px 8px}}
  .cat-btn{{display:flex;align-items:center;justify-content:space-between;width:100%;
            background:none;border:none;color:#aaa;font-size:13px;padding:8px 12px;
            text-align:left;cursor:pointer;border-left:2px solid transparent;
            transition:color .1s,background .1s}}
  .cat-btn:hover{{background:#1c1c1f;color:#e0e0e0}}
  .cat-btn.active{{color:#6db3f2;border-left-color:#6db3f2;background:#1a1f2a;font-weight:600}}
  .cat-btn .badge{{background:#252528;color:#888;font-size:11px;padding:1px 7px;border-radius:10px}}
  .cat-btn.active .badge{{background:#1e3050;color:#6db3f2}}
  main{{flex:1;overflow-y:auto;padding:16px}}
  .section-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
                  color:#888;margin:20px 0 10px;padding-bottom:7px;border-bottom:1px solid #252528;
                  display:flex;align-items:center;gap:8px}}
  .section-title:first-child{{margin-top:0}}
  .section-title .scount{{color:#555;font-size:11px;font-weight:400;text-transform:none;letter-spacing:0}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-bottom:6px}}
  .card{{background:#181820;border:1px solid #2a2a30;border-radius:8px;overflow:hidden;
         cursor:pointer;transition:border-color .15s,transform .12s,box-shadow .15s}}
  .card:hover{{border-color:#4a8fd4;transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.5)}}
  .card-img{{width:100%;aspect-ratio:1;object-fit:cover;background:#0e0e10;display:block}}
  .card-img-ph{{width:100%;aspect-ratio:1;background:#111113;display:flex;align-items:center;
                justify-content:center;color:#333;font-size:32px}}
  .card-body{{padding:9px 10px}}
  .card-name{{font-size:13px;font-weight:600;color:#e8e8e8;white-space:nowrap;overflow:hidden;
              text-overflow:ellipsis;margin-bottom:4px}}
  .card-cat{{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}}
  .card-tags{{font-size:11px;color:#6a90b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  #overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:100;
            align-items:center;justify-content:center;padding:20px}}
  #overlay.open{{display:flex}}
  .detail{{background:#1a1a1d;border:1px solid #2e2e34;border-radius:10px;max-width:700px;
           width:100%;display:flex;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.7)}}
  .detail-img-wrap{{width:320px;flex-shrink:0;position:relative;background:#0e0e10}}
  .detail-img{{width:100%;height:100%;object-fit:cover;background:#0e0e10;display:block}}
  .detail-img-ph{{width:320px;flex-shrink:0;background:#111113;display:flex;align-items:center;
                  justify-content:center;color:#2a2a2e;font-size:56px}}
  .render-strip{{position:absolute;bottom:0;left:0;right:0;display:flex;gap:3px;padding:5px;
                 background:linear-gradient(transparent,rgba(0,0,0,.75));pointer-events:none}}
  .render-thumb{{width:52px;height:52px;object-fit:cover;border-radius:3px;cursor:pointer;
                 opacity:.6;border:2px solid transparent;transition:opacity .15s,border-color .15s;
                 pointer-events:auto}}
  .render-thumb:hover{{opacity:.9}}
  .render-thumb.active{{opacity:1;border-color:#6db3f2}}
  .detail-info{{padding:24px 22px;flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:12px}}
  .detail-name{{font-size:18px;font-weight:700;color:#f0f0f0;line-height:1.3}}
  .detail-cat{{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#6db3f2;font-weight:600}}
  .detail-sep{{border:none;border-top:1px solid #2a2a30}}
  .detail-label{{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:#777;margin-bottom:4px}}
  .detail-val{{font-size:13px;color:#c8c8c8;line-height:1.6}}
  .tag-list{{display:flex;flex-wrap:wrap;gap:6px}}
  .tag{{background:#1e2a3a;color:#7ab0d8;font-size:12px;padding:3px 9px;border-radius:4px;
        border:1px solid #2a3f55}}
  .detail-meta{{font-size:12px;color:#666;margin-top:auto;padding-top:10px;border-top:1px solid #222}}
  .close-btn{{position:absolute;top:12px;right:14px;background:#252528;border:none;color:#aaa;
              font-size:16px;width:28px;height:28px;border-radius:50%;cursor:pointer;
              display:flex;align-items:center;justify-content:center;transition:background .1s}}
  .close-btn:hover{{background:#333;color:#eee}}
  ::-webkit-scrollbar{{width:6px;height:6px}}
  ::-webkit-scrollbar-track{{background:transparent}}
  ::-webkit-scrollbar-thumb{{background:#2a2a2e;border-radius:3px}}
</style>
</head>
<body>
<header>
  <h1>Asset <span>Library</span></h1>
  <input id="search" type="text" placeholder="Search name, tags, description…" oninput="applyFilter()">
  <span id="count-label">{count} assets</span>
  <span id="gen-time">Generated {timestamp}</span>
</header>
<div class="body">
  <aside id="sidebar"><h2>Categories</h2><div id="cat-list"></div></aside>
  <main id="main"><div id="grid-root"></div></main>
</div>
<div id="overlay" onclick="closeDetail(event)">
  <div class="detail" id="detail-panel" style="position:relative">
    <button class="close-btn" onclick="closeDetail(null)">✕</button>
    <div class="detail-img-wrap" id="d-img-wrap" style="display:none">
      <img id="d-img" class="detail-img">
      <div class="render-strip" id="d-strip"></div>
    </div>
    <div id="d-img-ph" class="detail-img-ph" style="display:none">📦</div>
    <div class="detail-info">
      <div class="detail-cat" id="d-cat"></div>
      <div class="detail-name" id="d-name"></div>
      <hr class="detail-sep">
      <div id="d-desc-block" style="display:none">
        <div class="detail-label">Description</div>
        <div class="detail-val" id="d-desc"></div>
      </div>
      <div id="d-tags-block" style="display:none">
        <div class="detail-label">Tags</div>
        <div class="tag-list" id="d-tags"></div>
      </div>
      <div>
        <div class="detail-label">Object</div>
        <div class="detail-val" id="d-object"></div>
      </div>
      <div class="detail-meta" id="d-meta"></div>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
let active='All', query='';
function cap(s){{return s?s[0].toUpperCase()+s.slice(1):s}}
function buildSidebar(){{
  const counts={{}};
  DATA.forEach(a=>{{counts[a.category]=(counts[a.category]||0)+1}});
  const cats=['All',...Object.keys(counts).sort()];
  const list=document.getElementById('cat-list');
  list.innerHTML='';
  cats.forEach(cat=>{{
    const n=cat==='All'?DATA.length:counts[cat];
    const btn=document.createElement('button');
    btn.className='cat-btn'+(cat===active?' active':'');
    btn.innerHTML=cap(cat)+' <span class="badge">'+n+'</span>';
    btn.onclick=()=>{{active=cat;buildSidebar();applyFilter()}};
    list.appendChild(btn);
  }});
}}
function applyFilter(){{
  query=document.getElementById('search').value.toLowerCase().trim();
  let f=DATA.filter(a=>{{
    if(active!=='All'&&a.category!==active)return false;
    if(!query)return true;
    return a.name.toLowerCase().includes(query)||a.tags.toLowerCase().includes(query)
          ||a.description.toLowerCase().includes(query)||a.object.toLowerCase().includes(query);
  }});
  document.getElementById('count-label').textContent=f.length+' asset'+(f.length!==1?'s':'');
  const root=document.getElementById('grid-root');
  const groups={{}};
  f.forEach(a=>{{(groups[a.category]=groups[a.category]||[]).push(a)}});
  root.innerHTML='';
  const showH=active==='All';
  Object.keys(groups).sort().forEach(cat=>{{
    if(showH){{
      const h=document.createElement('div');
      h.className='section-title';
      h.innerHTML=cap(cat)+' <span class="scount">'+groups[cat].length+' assets</span>';
      root.appendChild(h);
    }}
    const grid=document.createElement('div');
    grid.className='grid';
    groups[cat].forEach(a=>grid.appendChild(makeCard(a)));
    root.appendChild(grid);
  }});
  if(!f.length)root.innerHTML='<div style="color:#444;padding:40px;text-align:center">No assets match.</div>';
}}
function makeCard(a){{
  const card=document.createElement('div');
  card.className='card';
  card.onclick=()=>openDetail(a);
  const img=a.img_b64
    ?'<img class="card-img" src="data:image/jpeg;base64,'+a.img_b64+'" loading="lazy">'
    :'<div class="card-img-ph">📦</div>';
  const tags=a.tags?a.tags.split(',').slice(0,3).map(t=>t.trim()).join('  ·  '):'';
  card.innerHTML=img+'<div class="card-body">'
    +'<div class="card-name" title="'+a.name+'">'+a.name.replace(/_/g,' ')+'</div>'
    +'<div class="card-cat">'+a.category+'</div>'
    +(tags?'<div class="card-tags">'+tags+'</div>':'')
    +'</div>';
  return card;
}}
function setDetailImg(b64){{
  document.getElementById('d-img').src='data:image/jpeg;base64,'+b64;
}}
function openDetail(a){{
  // Show/hide main image wrapper vs placeholder
  const wrap=document.getElementById('d-img-wrap'),ph=document.getElementById('d-img-ph');
  if(a.img_b64){{wrap.style.display='';ph.style.display='none';setDetailImg(a.img_b64);}}
  else{{wrap.style.display='none';ph.style.display='';}}
  // Build render strip (small thumbnails overlaid at bottom of main image)
  const strip=document.getElementById('d-strip');
  strip.innerHTML='';
  const renders=[
    {{b64:a.img_b64,      label:'3/4'}},
    {{b64:a.img_front_b64,label:'Front'}},
    {{b64:a.img_side_b64, label:'Side'}},
  ].filter(r=>r.b64);
  if(renders.length>1){{
    renders.forEach((r,i)=>{{
      const t=document.createElement('img');
      t.className='render-thumb'+(i===0?' active':'');
      t.src='data:image/jpeg;base64,'+r.b64;
      t.title=r.label;
      t.onclick=()=>{{
        strip.querySelectorAll('.render-thumb').forEach(x=>x.classList.remove('active'));
        t.classList.add('active');
        setDetailImg(r.b64);
      }};
      strip.appendChild(t);
    }});
    strip.style.display='flex';
  }} else {{
    strip.style.display='none';
  }}
  document.getElementById('d-cat').textContent=cap(a.category);
  document.getElementById('d-name').textContent=a.name.replace(/_/g,' ');
  document.getElementById('d-object').textContent=a.object;
  const db=document.getElementById('d-desc-block');
  if(a.description){{document.getElementById('d-desc').textContent=a.description;db.style.display=''}}
  else db.style.display='none';
  const tb=document.getElementById('d-tags-block'),tl=document.getElementById('d-tags');
  if(a.tags){{tl.innerHTML=a.tags.split(',').map(t=>'<span class="tag">'+t.trim()+'</span>').join('');tb.style.display=''}}
  else tb.style.display='none';
  const m=[];
  if(a.date)m.push(a.date);if(a.res)m.push(a.res+'px');
  if(a.frames)m.push(a.frames+' frames');if(a.nim)m.push('NIM recognized');
  document.getElementById('d-meta').textContent=m.join('  ·  ');
  document.getElementById('overlay').classList.add('open');
}}
function closeDetail(e){{
  if(e&&document.getElementById('detail-panel').contains(e.target))return;
  document.getElementById('overlay').classList.remove('open');
}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeDetail(null)}});
buildSidebar();applyFilter();
</script>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Scanning {ASSET_DIR} ...")
    assets = scan_assets(ASSET_DIR)
    print(f"Found {len(assets)} assets")

    if not assets:
        print("[warn] No assets found. Run batch processing first.")

    data_json = json.dumps(assets, ensure_ascii=False)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = HTML_TEMPLATE.format(
        count     = len(assets),
        timestamp = timestamp,
        data_json = data_json,
    )

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Written → {OUTPUT.resolve()}")

    # Auto-open in default browser
    webbrowser.open(OUTPUT.resolve().as_uri())


if __name__ == "__main__":
    main()
