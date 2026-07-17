"""
SatViz-style Cesium visualization of the OEC scenario.

Generates the same kind of 3D-globe page as Hypatia's satviz (Cesium 1.57,
white toner-style globe) but showing the *actual* simulation instead of
static orbit rings:

  * Kuiper-630 constellation animated over the full 5 h window
    (client-side Walker propagation, identical math to geometry.py)
  * +Grid ISLs gated per frame by the line-of-sight + max-range test
  * GBS markers with elevation-gated GSLs, AOI markers with coverage links
  * The MPC task schedule from oec_scenario/ overlaid: active tasks are
    drawn as shortest-delay routes (Dijkstra, as in topology.py) coloured
    by the depth the MPC chose, with a live stats HUD

Run:  python3 -m oec_sim.satviz          (from hypatia_sim/)
Out:  oec_scenario/satviz_oec.html       (open in a browser; internet is
      required for the Cesium library and map tiles)

The Cesium ion token is copied from hypatia/satviz/viz_output/kuiper_630.html
if present.
"""

import csv
import json
import os
import re

from . import config as C
from .topology import build_isl_pairs

OUT_FILE = os.path.join(C.OUT_DIR, 'satviz_oec.html')
_TOKEN_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'hypatia', 'satviz', 'viz_output', 'kuiper_630.html')


def _ion_token():
    try:
        with open(_TOKEN_SOURCE) as f:
            m = re.search(r"defaultAccessToken = '([^']+)'", f.read())
        if m and not m.group(1).startswith('<'):
            return m.group(1)
    except OSError:
        pass
    return ''


def _load_tasks():
    """tasks.csv joined with the MPC's chosen depth per task."""
    gs_idx = {g['name']: i for i, g in enumerate(C.GROUND_STATIONS)}
    depth, frac = {}, {}
    try:
        with open(os.path.join(C.OUT_DIR, 'task_outcomes_mpc.csv')) as f:
            for row in csv.DictReader(f):
                depth[int(row['kid'])] = int(row['depth'])
                frac[int(row['kid'])] = float(row['delivery_fraction'])
    except OSError:
        pass
    tasks = []
    try:
        with open(os.path.join(C.OUT_DIR, 'tasks.csv')) as f:
            for row in csv.DictReader(f):
                kid = int(row['kid'])
                tasks.append({
                    'kid': kid,
                    'aoi': row['aoi'],
                    'src': int(row['src_sat']),
                    'gs': gs_idx[row['dst_gs']],
                    'arrival': float(row['arrival_s']),
                    'deadline': float(row['deadline_s']),
                    'images': int(row['n_images']),
                    'weight': int(row['weight']),
                    'depth': depth.get(kid, 0),
                    'frac': frac.get(kid, 0.0),
                })
    except OSError:
        pass
    return tasks


def _load_timeline():
    tl = {'t': [], 'img': [], 'util': [], 'backlog': [], 'active': []}
    try:
        with open(os.path.join(C.OUT_DIR, 'timeline_mpc.csv')) as f:
            for row in csv.DictReader(f):
                tl['t'].append(float(row['t_s']))
                tl['img'].append(float(row['delivered_images']))
                tl['util'].append(float(row['utility']))
                tl['backlog'].append(float(row['backlog_bits']))
                tl['active'].append(int(row['n_active']))
    except OSError:
        pass
    return tl


def build_data():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'land_110m.json')) as f:
        land = json.load(f)
    return {
        'constellation': C.CONSTELLATION_NAME,
        'nPlanes': C.N_PLANES, 'satsPerPlane': C.SATS_PER_PLANE,
        'nSats': C.N_SATS,
        'altKm': C.ALT_KM, 'inclDeg': C.INCL_DEG,
        'aSma': C.A_SMA, 'inclRad': C.INCL_DEG * C.DEG,
        'phasingF': C.PHASING_F, 'meanMotion': C.N_MM,
        'omegaE': C.OMEGA_E, 'rE': C.R_E, 'cKmS': C.C_KM_S,
        'losGrazeKm': C.LOS_GRAZE_KM, 'islMaxKm': C.ISL_MAX_KM,
        'minElevDeg': C.MIN_ELEV_DEG, 'aoiElevDeg': C.AOI_ELEV_DEG,
        'simS': C.SIM_S, 'slotS': C.SLOT_S,
        'islRateBps': C.ISL_RATE_BPS, 'gsRateBps': C.GS_RATE_BPS,
        'depths': C.DEPTHS,
        'utility': {str(q): C.UTILITY[q] for q in C.DEPTHS},
        'payloadB': {str(q): C.PAYLOAD_B[q] for q in C.DEPTHS},
        'encMsPerImg': C.ENC_S_PER_IMG * 1e3,
        'mpcH': C.MPC_HORIZON_SLOTS, 'mpcResolve': C.MPC_RESOLVE_EVERY,
        'model': ('FLAIR-1 RQ-VAE — 512&times;512 aerial tiles &rarr; '
                  '8&times;8 latent grid, depth-16 checkpoint truncated '
                  'to the first q stages'),
        'gs': C.GROUND_STATIONS, 'aois': C.AOIS,
        'islPairs': build_isl_pairs().tolist(),
        'tasks': _load_tasks(),
        'timeline': _load_timeline(),
        'land': land,
    }


_HTML = r"""<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OEC SatViz — __CONSTELLATION__</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.57/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.57/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <style>
    html, body { margin: 0; padding: 0; height: 100%; }
    #hud {
      position: absolute; top: 8px; left: 8px; z-index: 10;
      background: rgba(255,255,255,0.92); border: 1px solid #999;
      border-radius: 4px; padding: 8px 10px;
      font: 12px/1.5 Menlo, monospace; color: #111; max-width: 340px;
    }
    #hud h3 { margin: 0 0 4px 0; font-size: 12px; }
    #hud label { margin-right: 8px; white-space: nowrap; }
    #hud .task { margin: 0; }
    #hud .sw { display: inline-block; width: 9px; height: 9px; margin-right: 3px; }
    #hud .ln { display: inline-block; width: 14px; height: 3px; margin: 0 3px 2px 0; }
    #hud .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; border: 1px solid #fff; }
    #hud details { margin: 4px 0; border-top: 1px solid #ddd; padding-top: 3px; }
    #hud summary { cursor: pointer; font-weight: bold; }
    #hud #legend span.item { display: inline-block; margin-right: 10px; white-space: nowrap; }
    #hud #params { color: #333; }
    #hud #params b { color: #111; }
    #hud #stats { border-top: 1px solid #ddd; margin-top: 4px; padding-top: 3px; }
  </style>
</head>
<body>
  <div id="cesiumContainer" style="width: 100%; height:100%"></div>
  <div id="hud">
    <h3>OEC RQ-NAC — __CONSTELLATION__ + MPC</h3>
    <div id="toggles">
      <label><input type="checkbox" id="tIsl" checked>ISLs</label>
      <label><input type="checkbox" id="tGsl" checked>GSLs</label>
      <label><input type="checkbox" id="tAoi" checked>AOI cover</label>
      <label><input type="checkbox" id="tRoute" checked>MPC routes</label>
    </div>
    <details open><summary>Legend</summary><div id="legend"></div></details>
    <details open><summary>Scenario &amp; model</summary><div id="params"></div></details>
    <div id="stats"></div>
    <div id="tasks"></div>
  </div>
  <script>
    Cesium.Ion.defaultAccessToken = '__TOKEN__';
    var viewer = new Cesium.Viewer('cesiumContainer', {
      skyBox: false,
      skyAtmosphere: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      shouldAnimate: true,
      contextOptions: { webgl: { alpha: true } }
    });

    var scene = viewer.scene;
    scene.backgroundColor = Cesium.Color.WHITE;
    scene.highDynamicRange = false;
    var globe = scene.globe;
    globe.imageryLayers.removeAll();
    globe.baseColor = Cesium.Color.fromCssColorString('#fdfdfd');

    var D = __DATA__;

    // Landmass is embedded as Natural Earth 110m polygons (no tile servers:
    // openstreetmap.org blocks file:// pages with "Access blocked" tiles, and
    // remote basemaps break offline). This also matches Hypatia's toner look.
    var landFill = Cesium.Color.fromCssColorString('#dde3e8');
    var coastCol = Cesium.Color.fromCssColorString('#9aa4ad');
    D.land.forEach(function(poly) {
      viewer.entities.add({ polygon: {
        hierarchy: new Cesium.PolygonHierarchy(
          Cesium.Cartesian3.fromDegreesArray(poly.o),
          poly.h.map(function(hole) {
            return new Cesium.PolygonHierarchy(
              Cesium.Cartesian3.fromDegreesArray(hole));
          })),
        material: landFill
      }});
      viewer.entities.add({ polyline: {
        positions: Cesium.Cartesian3.fromDegreesArray(poly.o),
        width: 1, material: coastCol, clampToGround: false
      }});
    });
    var N = D.nSats, PAIRS = D.islPairs, GS = D.gs, AOIS = D.aois;

    // ── Walker propagation (mirrors oec_sim/geometry.py) ─────────────────
    var raan = new Float64Array(N), anom0 = new Float64Array(N);
    for (var p = 0; p < D.nPlanes; p++)
      for (var s = 0; s < D.satsPerPlane; s++) {
        var i = p * D.satsPerPlane + s;
        raan[i] = p * 2 * Math.PI / D.nPlanes;
        anom0[i] = s * 2 * Math.PI / D.satsPerPlane
                 + p * D.phasingF * 2 * Math.PI / N;
      }
    var ci = Math.cos(D.inclRad), si = Math.sin(D.inclRad);
    var px = new Float64Array(N), py = new Float64Array(N),
        pz = new Float64Array(N);                       // km, ECEF
    function propagate(t) {
      var g = D.omegaE * t, cg = Math.cos(g), sg = Math.sin(g);
      for (var i = 0; i < N; i++) {
        var th = anom0[i] + D.meanMotion * t;
        var xo = D.aSma * Math.cos(th), yo = D.aSma * Math.sin(th);
        var cr = Math.cos(raan[i]), sr = Math.sin(raan[i]);
        var xe = cr * xo - sr * ci * yo;
        var ye = sr * xo + cr * ci * yo;
        px[i] =  cg * xe + sg * ye;
        py[i] = -sg * xe + cg * ye;
        pz[i] = si * yo;
      }
    }
    function cart(i) {
      return new Cesium.Cartesian3(px[i] * 1e3, py[i] * 1e3, pz[i] * 1e3);
    }

    // ground points: ECEF position + unit normal (sphere, as in geometry.py)
    function groundEcef(pt) {
      var la = pt.lat * Math.PI / 180, lo = pt.lon * Math.PI / 180;
      var n = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo),
               Math.sin(la)];
      return { n: n, p: [n[0] * D.rE, n[1] * D.rE, n[2] * D.rE],
               c: new Cesium.Cartesian3(n[0] * D.rE * 1e3, n[1] * D.rE * 1e3,
                                        n[2] * D.rE * 1e3) };
    }
    var gsPts = GS.map(groundEcef), aoiPts = AOIS.map(groundEcef);

    function elevDeg(i, gp) {
      var dx = px[i] - gp.p[0], dy = py[i] - gp.p[1], dz = pz[i] - gp.p[2];
      var dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      var sinEl = (dx * gp.n[0] + dy * gp.n[1] + dz * gp.n[2]) / dist;
      return { el: Math.asin(Math.max(-1, Math.min(1, sinEl))) * 180 / Math.PI,
               dist: dist };
    }

    // ISL feasibility: closest approach above R_E + graze, length <= max
    function islCheck(a, b) {
      var dx = px[b] - px[a], dy = py[b] - py[a], dz = pz[b] - pz[a];
      var len2 = dx * dx + dy * dy + dz * dz, len = Math.sqrt(len2);
      if (len > D.islMaxKm) return 0;
      var ts = -(px[a] * dx + py[a] * dy + pz[a] * dz) / Math.max(len2, 1e-9);
      ts = Math.min(Math.max(ts, 0), 1);
      var cx = px[a] + ts * dx, cy = py[a] + ts * dy, cz = pz[a] + ts * dz;
      if (Math.sqrt(cx * cx + cy * cy + cz * cz) < D.rE + D.losGrazeKm)
        return 0;
      return len;
    }

    // ── primitives ───────────────────────────────────────────────────────
    var satPoints = scene.primitives.add(new Cesium.PointPrimitiveCollection());
    for (var i = 0; i < N; i++)
      satPoints.add({ position: Cesium.Cartesian3.ZERO, pixelSize: 3,
                      color: Cesium.Color.BLACK });

    var markers = scene.primitives.add(new Cesium.PointPrimitiveCollection());
    var labels = scene.primitives.add(new Cesium.LabelCollection());
    function addMarker(pt, ecef, color) {
      markers.add({ position: ecef.c, pixelSize: 7, color: color,
                    outlineColor: Cesium.Color.WHITE, outlineWidth: 1 });
      labels.add({ position: ecef.c, text: pt.name,
                   font: '11px sans-serif',
                   fillColor: Cesium.Color.fromCssColorString('#111'),
                   pixelOffset: new Cesium.Cartesian2(8, -8),
                   horizontalOrigin: Cesium.HorizontalOrigin.LEFT });
    }
    GS.forEach(function(g, k) {
      addMarker(g, gsPts[k], Cesium.Color.CRIMSON); });
    AOIS.forEach(function(a, k) {
      addMarker(a, aoiPts[k], Cesium.Color.DARKORANGE); });

    function colorMat(css, alpha) {
      return Cesium.Material.fromType('Color', {
        color: Cesium.Color.fromCssColorString(css).withAlpha(alpha) });
    }
    var islLines = scene.primitives.add(new Cesium.PolylineCollection());
    for (var l = 0; l < PAIRS.length; l++)
      islLines.add({ positions: [], width: 1, show: false,
                     material: colorMat('#1e90ff', 0.35) });   // dodgerblue

    function makePool(coll, n, css, alpha, width) {
      var pool = [];
      for (var k = 0; k < n; k++)
        pool.push(coll.add({ positions: [], width: width, show: false,
                             material: colorMat(css, alpha) }));
      return pool;
    }
    var linkColl = scene.primitives.add(new Cesium.PolylineCollection());
    var gslPool = makePool(linkColl, 120, '#dc143c', 0.5, 1.5);  // crimson
    var aoiPool = makePool(linkColl, 120, '#ff8c00', 0.5, 1.5);  // darkorange

    var DEPTH_CSS = { 0: '#7f8c8d', 1: '#27ae60', 2: '#f1c40f',
                      4: '#e67e22', 8: '#c0392b', 16: '#8e44ad' };
    var routeColl = scene.primitives.add(new Cesium.PolylineCollection());
    var routePool = [];
    for (var k = 0; k < 16; k++) {
      var pl = routeColl.add({ positions: [], width: 3, show: false,
                               material: colorMat('#7f8c8d', 0.9) });
      pl._depth = -1;
      routePool.push(pl);
    }

    // ── shortest-delay routing (mirrors topology.py: Dijkstra, km weights) ─
    var adj = [];        // node -> [v, w, v, w, ...]; GBS g is node N + g
    function buildAdj(islLen) {
      adj = new Array(N + GS.length);
      for (var i = 0; i < adj.length; i++) adj[i] = [];
      for (var l = 0; l < PAIRS.length; l++) {
        if (!islLen[l]) continue;
        var a = PAIRS[l][0], b = PAIRS[l][1];
        adj[a].push(b, islLen[l]); adj[b].push(a, islLen[l]);
      }
      for (var g = 0; g < GS.length; g++)
        for (var i2 = 0; i2 < N; i2++) {
          var e = elevDeg(i2, gsPts[g]);
          if (e.el >= D.minElevDeg) {
            adj[i2].push(N + g, e.dist); adj[N + g].push(i2, e.dist);
          }
        }
    }
    function dijkstra(gsIdx) {
      var n = N + GS.length;
      var dist = new Float64Array(n).fill(Infinity);
      var pred = new Int32Array(n).fill(-1);
      var done = new Uint8Array(n);
      var src = N + gsIdx;
      dist[src] = 0;
      var heap = [[0, src]];
      while (heap.length) {
        // binary-heap pop
        var top = heap[0], last = heap.pop();
        if (heap.length) {
          heap[0] = last;
          for (var j = 0;;) {
            var c = 2 * j + 1;
            if (c >= heap.length) break;
            if (c + 1 < heap.length && heap[c + 1][0] < heap[c][0]) c++;
            if (heap[c][0] >= heap[j][0]) break;
            var tmp = heap[j]; heap[j] = heap[c]; heap[c] = tmp; j = c;
          }
        }
        var u = top[1];
        if (done[u]) continue;
        done[u] = 1;
        var es = adj[u];
        for (var e2 = 0; e2 < es.length; e2 += 2) {
          var v = es[e2], nd = top[0] + es[e2 + 1];
          // as in topology.py: only this GBS's GSL edges exist, so routes
          // never transit through a different ground station
          if (v >= N && v !== src) continue;
          if (nd < dist[v]) {
            dist[v] = nd; pred[v] = u;
            // binary-heap push
            heap.push([nd, v]);
            for (var j2 = heap.length - 1; j2 > 0;) {
              var par = (j2 - 1) >> 1;
              if (heap[par][0] <= heap[j2][0]) break;
              var t2 = heap[par]; heap[par] = heap[j2]; heap[j2] = t2;
              j2 = par;
            }
          }
        }
      }
      return { dist: dist, pred: pred };
    }
    function walkPath(pred, gsIdx, src) {
      var gsNode = N + gsIdx, path = [src], node = src;
      for (var hop = 0; hop < 200 && node !== gsNode; hop++) {
        node = pred[node];
        if (node < 0) return null;
        path.push(node);
      }
      return node === gsNode ? path : null;
    }

    // ── HUD: legend + scenario/model info ────────────────────────────────
    function sw(css) { return '<span class="sw" style="background:' + css + '"></span>'; }
    function ln(css) { return '<span class="ln" style="background:' + css + '"></span>'; }
    function dot(css) { return '<span class="dot" style="background:' + css + '"></span>'; }
    var legendHtml =
        '<span class="item">' + dot('#000') + 'satellite</span>'
      + '<span class="item">' + dot('#dc143c') + 'ground station (GBS)</span>'
      + '<span class="item">' + dot('#ff8c00') + 'area of interest (AOI)</span>'
      + '<br><span class="item">' + ln('#1e90ff') + 'ISL (line-of-sight ok, &le;'
      + D.islMaxKm.toLocaleString() + ' km)</span>'
      + '<br><span class="item">' + ln('#dc143c') + 'GSL (elev &ge; '
      + D.minElevDeg + '&deg;)</span>'
      + '<span class="item">' + ln('#ff8c00') + 'AOI in view (elev &ge; '
      + D.aoiElevDeg + '&deg;)</span>'
      + '<br>MPC route, colour = chosen depth q:<br>'
      + D.depths.map(function(q) {
          return '<span class="item">' + ln(DEPTH_CSS[q]) + 'q=' + q + '</span>';
        }).join('')
      + '<span class="item">' + ln('#7f8c8d') + 'unscheduled</span>';
    document.getElementById('legend').innerHTML = legendHtml;

    var paramsHtml =
        '<b>Compression model:</b> ' + D.model + '.<br>'
      + '<b>Utility u<sub>q</sub></b> (1&minus;LPIPS, measured): '
      + D.depths.map(function(q) {
          return 'q=' + q + ': ' + D.utility[q].toFixed(3); }).join(', ')
      + '.<br><b>Payload:</b> 88&middot;q B/img &mdash; '
      + D.depths.map(function(q) {
          return D.payloadB[q] + ' B'; }).join(' / ')
      + '; <b>encoder:</b> ' + D.encMsPerImg.toFixed(2) + ' ms/img.<br>'
      + '<b>Constellation:</b> ' + D.constellation + ' &mdash; Walker '
      + D.nSats + '/' + D.nPlanes + '/' + D.phasingF + ', '
      + D.altKm + ' km, ' + D.inclDeg + '&deg;, +Grid.<br>'
      + '<b>Links:</b> ISL ' + (D.islRateBps / 1e6) + ' Mbps; GBS '
      + (D.gsRateBps / 1e6) + ' Mbps aggregate.<br>'
      + '<b>Scheduler:</b> rolling-horizon MPC (MILP/HiGHS), H = '
      + D.mpcH + '&times;' + D.slotS + ' s = '
      + (D.mpcH * D.slotS / 60) + ' min, replans every &le;'
      + D.mpcResolve + ' slots and on arrivals.<br>'
      + '<b>Routing:</b> shortest-delay Dijkstra per GBS over the feasible '
      + 'graph (no relaying via other GBSs).';
    document.getElementById('params').innerHTML = paramsHtml;

    var showIsl = true, showGsl = true, showAoi = true, showRoute = true;
    function bind(id, fn) {
      document.getElementById(id).addEventListener('change', function(ev) {
        fn(ev.target.checked); lastHeavyT = -1e9;   // force refresh
      });
    }
    bind('tIsl', function(v) { showIsl = v; });
    bind('tGsl', function(v) { showGsl = v; });
    bind('tAoi', function(v) { showAoi = v; });
    bind('tRoute', function(v) { showRoute = v; });

    function fmtT(t) {
      var h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60),
          s = Math.floor(t % 60);
      return h + ':' + ('0' + m).slice(-2) + ':' + ('0' + s).slice(-2);
    }
    var tlIdx = 0;
    function timelineAt(t) {
      var T = D.timeline.t;
      if (!T.length) return null;
      while (tlIdx + 1 < T.length && T[tlIdx + 1] <= t) tlIdx++;
      while (tlIdx > 0 && T[tlIdx] > t) tlIdx--;
      return tlIdx;
    }

    // ── per-frame + throttled heavy refresh ──────────────────────────────
    var start = Cesium.JulianDate.fromIso8601('2000-01-01T00:00:00Z');
    var stop = Cesium.JulianDate.addSeconds(start, D.simS,
                                            new Cesium.JulianDate());
    viewer.clock.startTime = start.clone();
    viewer.clock.stopTime = stop;
    viewer.clock.currentTime = start.clone();
    viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
    viewer.clock.multiplier = 60;
    viewer.timeline.zoomTo(start, stop);

    var lastHeavy = 0, lastHeavyT = -1e9;
    var islLen = new Float64Array(PAIRS.length);

    function heavy(t) {
      // ISLs
      var islCount = 0;
      for (var l = 0; l < PAIRS.length; l++) {
        islLen[l] = islCheck(PAIRS[l][0], PAIRS[l][1]);
        var line = islLines.get(l);
        var vis = showIsl && islLen[l] > 0;
        line.show = vis;
        if (vis) line.positions = [cart(PAIRS[l][0]), cart(PAIRS[l][1])];
        if (islLen[l] > 0) islCount++;
      }
      // GSLs
      var visCount = [], used = 0;
      for (var g = 0; g < GS.length; g++) {
        visCount[g] = 0;
        for (var i = 0; i < N; i++) {
          if (elevDeg(i, gsPts[g]).el >= D.minElevDeg) {
            visCount[g]++;
            if (showGsl && used < gslPool.length) {
              var pl = gslPool[used++];
              pl.show = true; pl.positions = [cart(i), gsPts[g].c];
            }
          }
        }
      }
      for (; used < gslPool.length; used++) gslPool[used].show = false;
      // AOI coverage
      used = 0;
      for (var a = 0; a < AOIS.length; a++)
        for (var i3 = 0; i3 < N; i3++) {
          if (elevDeg(i3, aoiPts[a]).el >= D.aoiElevDeg) {
            if (showAoi && used < aoiPool.length) {
              var pl2 = aoiPool[used++];
              pl2.show = true; pl2.positions = [cart(i3), aoiPts[a].c];
            }
          }
        }
      for (; used < aoiPool.length; used++) aoiPool[used].show = false;
      // MPC task routes
      var active = D.tasks.filter(function(k) {
        return t >= k.arrival && t <= k.deadline; });
      var taskHtml = '';
      used = 0;
      if (active.length) {
        buildAdj(islLen);
        var byGs = {};
        active.forEach(function(k) {
          (byGs[k.gs] = byGs[k.gs] || []).push(k); });
        Object.keys(byGs).forEach(function(g2) {
          var res = dijkstra(+g2);
          byGs[g2].forEach(function(k) {
            var path = walkPath(res.pred, +g2, k.src);
            var info = '(no route)';
            if (path && used < routePool.length) {
              var pl3 = routePool[used++];
              pl3.show = showRoute;
              pl3.positions = path.map(function(n) {
                return n < N ? cart(n) : gsPts[n - N].c; });
              if (pl3._depth !== k.depth) {
                pl3.material = colorMat(DEPTH_CSS[k.depth] || '#7f8c8d', 0.9);
                pl3._depth = k.depth;
              }
              info = (path.length - 1) + ' hops, '
                   + (res.dist[k.src] / D.cKmS * 1e3).toFixed(1) + ' ms';
            }
            taskHtml += '<p class="task"><span class="sw" style="background:'
              + (DEPTH_CSS[k.depth] || '#7f8c8d') + '"></span>#' + k.kid
              + ' ' + k.aoi + ' &rarr; ' + GS[k.gs].name
              + ' q=' + k.depth + ' ' + info + '</p>';
          });
        });
      }
      for (; used < routePool.length; used++) routePool[used].show = false;
      // stats
      var slot = Math.floor(t / D.slotS);
      var html = 't = ' + fmtT(t) + ' (slot ' + slot + '/'
               + Math.floor(D.simS / D.slotS) + ')<br>'
               + 'feasible ISLs: ' + islCount + ' / ' + PAIRS.length + '<br>';
      for (var g3 = 0; g3 < GS.length; g3++)
        html += GS[g3].name + ': ' + visCount[g3] + ' sats  ';
      var ti = timelineAt(t);
      if (ti !== null)
        html += '<br>MPC: ' + Math.round(D.timeline.img[ti]).toLocaleString()
             + ' img delivered, utility ' + D.timeline.util[ti].toFixed(3)
             + ',<br>backlog ' + (D.timeline.backlog[ti] / 8e6).toFixed(0)
             + ' MB, ' + D.timeline.active[ti] + ' active tasks';
      document.getElementById('stats').innerHTML = html;
      document.getElementById('tasks').innerHTML = taskHtml;
    }

    viewer.clock.onTick.addEventListener(function(clock) {
      var t = Cesium.JulianDate.secondsDifference(clock.currentTime, start);
      propagate(t);
      for (var i = 0; i < N; i++) satPoints.get(i).position = cart(i);
      var now = performance.now();
      if (now - lastHeavy > 250 || Math.abs(t - lastHeavyT) > 120) {
        lastHeavy = now; lastHeavyT = t;
        heavy(t);
      }
    });

    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(19.57, 10.32, 20000000.0),
      duration: 0
    });
    viewer.resolutionScale = window.devicePixelRatio;
  </script>
</body>
</html>
"""


def main():
    data = build_data()
    html = (_HTML
            .replace('__CONSTELLATION__', data['constellation'])
            .replace('__TOKEN__', _ion_token())
            .replace('__DATA__', json.dumps(data, separators=(',', ':'))))
    os.makedirs(C.OUT_DIR, exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        f.write(html)
    print('wrote', OUT_FILE,
          '(%d tasks, %d ISL pairs)' % (len(data['tasks']),
                                        len(data['islPairs'])))


if __name__ == '__main__':
    main()
