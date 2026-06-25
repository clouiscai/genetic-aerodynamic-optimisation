"""Emit an interactive ForgeCAD "genetics lab" design.

Instead of hand-sliders for every gene (there are 4 × ``num_sections`` per-section
genes), this exposes the *evolutionary* controls:

* **Parent A / Parent B seed** — each integer seed produces a complete random
  genome sampled within the **same bounds** as ``genome.random_genome`` (per-section
  slope random-walks, per-section pitch/scale, winglet ellipse genes, pointy/flat).
* **Crossover seed** — performs **random assortment**: each gene *group* (airfoil,
  pitch, sweep, dihedral, scale, winglet) is inherited wholesale from a random parent.
* **Mutation strength** — optional per-gene jitter (scaled by each gene's range).
* **Show** — view the assorted child, or either parent, to compare.

Drag the seeds to watch how random assortment recombines two wings. The decode /
geometry math mirrors ``genome.py`` + ``geometry_generator.py`` exactly.
"""

from __future__ import annotations

import json

from .airfoils import normalized_airfoil_population
from .config import GeneratorConfig


_TEMPLATE = r"""
// =====================================================================
// Interactive GENETICS LAB — random assortment of wing genomes.
// Two parent seeds each generate a full random genome within the config bounds
// (per-section slope random-walks, pitch/scale rates, winglet ellipse genes).
// The crossover seed assorts gene-groups into a child; mutation jitters genes.
// Wing sweep/dihedral are integrated from per-section SLOPES (so the curve may
// carry several inflection points). Frame: X = chord, Y = span, Z = vertical up.
// =====================================================================

const AIRFOILS = __AIRFOILS_JSON__;
const AIRFOIL_NAMES = __AIRFOIL_NAMES_JSON__;

const MM = 1000.0;
const DEG = Math.PI / 180.0;

const N = __NUM_SECTIONS__;
const TOTAL_SPAN_M = __TOTAL_SPAN__;
const ROOT_CHORD_M = __ROOT_CHORD__;
const SCALE_MIN = __SCALE_MIN__;
const TWIST_LIMIT = __TWIST_MAX__;
const WINGTIP_SECTIONS = __WINGTIP_SECTIONS__;
const MIN_TIP_CHORD_M = __MIN_TIP_CHORD_M__;
const MIN_TIP_THICKNESS_M = __MIN_TIP_THICKNESS_M__;
const TIP_THICKNESS_M = __TIP_THICKNESS_M__;
const MAX_DSLOPE = __MAX_DSLOPE__;
const POINTY_PROB = __POINTY_PROB__;

// Gene bounds (mirror config.GeneBounds): [min, max]
const B = {
  rootPitch:  [__RP_MIN__, __RP_MAX__],
  hSlope:     [__HS_MIN__, __HS_MAX__],
  vSlope:     [__VS_MIN__, __VS_MAX__],
  pitchChange:[__PC_MIN__, __PC_MAX__],
  scaleDown:  [__SD_MIN__, __SD_MAX__],
  wLen:       [__WLEN_MIN__, __WLEN_MAX__],
  wRad:       [__WRAD_MIN__, __WRAD_MAX__],
  wRot:       [__WROT_MIN__, __WROT_MAX__],
  wArc:       [__WARC_MIN__, __WARC_MAX__],
  wHRate:     [__WHR_MIN__, __WHR_MAX__],
  wSRate:     [__WSR_MIN__, __WSR_MAX__],
  wFlatTip:   [__WFTC_MIN__, __WFTC_MAX__],
  wPointyX:   [__WPTX_MIN__, __WPTX_MAX__],
};

// ---------------------------------------------------------------- controls
const seedA   = Param.number("Parent A seed", 1, { min: 1, max: 9999 });
const seedB   = Param.number("Parent B seed", 2, { min: 1, max: 9999 });
const xoSeed  = Param.number("Crossover (assortment) seed", 7, { min: 1, max: 9999 });
const mutAmt  = Param.number("Mutation strength", 0.0, { min: 0.0, max: 1.0 });
const showWho = Param.choice("Show", "Child (assorted)", ["Child (assorted)", "Parent A", "Parent B"]);

// ---------------------------------------------------------------- prng + helpers
function mulberry32(a) {
  a = a >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function uni(rng, b) { return b[0] + (b[1] - b[0]) * rng(); }
function randomWalk(rng, n, b, maxStep) {
  let v = uni(rng, b);
  const out = [v];
  for (let i = 1; i < n; i++) { v = clamp(v + (rng() * 2 - 1) * maxStep, b[0], b[1]); out.push(v); }
  return out;
}
function smoothstep(t) { return 3.0 * t * t - 2.0 * t * t * t; }
function rotate2d(x, z, deg) { const a = deg * DEG, c = Math.cos(a), s = Math.sin(a); return [x * c - z * s, x * s + z * c]; }
function profileThickness(coords) {
  let mn = Infinity, mx = -Infinity;
  for (const p of coords) { mn = Math.min(mn, p[1]); mx = Math.max(mx, p[1]); }
  return Math.max(mx - mn, 1e-9);
}

// ---------------------------------------------------------------- genome (genes)
// One full random genome from a seed — matches genome.random_genome().
function buildGenome(seed) {
  const rng = mulberry32(seed);
  const pc = [0.0]; for (let i = 1; i < N; i++) pc.push(uni(rng, B.pitchChange));
  const sd = [0.0]; for (let i = 1; i < N; i++) sd.push(uni(rng, B.scaleDown));
  return {
    airfoil: AIRFOIL_NAMES[Math.floor(rng() * AIRFOIL_NAMES.length)],
    rootPitch: uni(rng, B.rootPitch),
    hSlope: randomWalk(rng, N, B.hSlope, MAX_DSLOPE),
    vSlope: randomWalk(rng, N, B.vSlope, MAX_DSLOPE),
    pitchChange: pc,
    scaleDown: sd,
    isPointy: rng() < POINTY_PROB,
    wLen: uni(rng, B.wLen), wRad: uni(rng, B.wRad), wRot: uni(rng, B.wRot),
    wArc: uni(rng, B.wArc), wHRate: uni(rng, B.wHRate), wSRate: uni(rng, B.wSRate),
    wFlatTip: uni(rng, B.wFlatTip), wPointyX: uni(rng, B.wPointyX),
  };
}

// Random assortment: each gene-GROUP comes wholesale from a random parent.
function crossover(A, C, seed) {
  const r = mulberry32(seed);
  const fromA = { airfoil: r() < 0.5, pitch: r() < 0.5, horiz: r() < 0.5, vert: r() < 0.5, scale: r() < 0.5, winglet: r() < 0.5 };
  const pick = (flag, a, b) => (flag ? a : b);
  const child = {
    airfoil: pick(fromA.airfoil, A.airfoil, C.airfoil),
    rootPitch: pick(fromA.pitch, A.rootPitch, C.rootPitch),
    pitchChange: pick(fromA.pitch, A.pitchChange, C.pitchChange),
    hSlope: pick(fromA.horiz, A.hSlope, C.hSlope),
    vSlope: pick(fromA.vert, A.vSlope, C.vSlope),
    scaleDown: pick(fromA.scale, A.scaleDown, C.scaleDown),
  };
  const wkeys = ["isPointy", "wLen", "wRad", "wRot", "wArc", "wHRate", "wSRate", "wFlatTip", "wPointyX"];
  for (const k of wkeys) child[k] = pick(fromA.winglet, A[k], C[k]);
  return child;
}

// Optional mutation: jitter each numeric gene by a fraction of its range.
function mutate(genome, amount, seed) {
  if (amount <= 0) return genome;
  const r = mulberry32(seed);
  const jit = (v, b) => clamp(v + (r() * 2 - 1) * amount * (b[1] - b[0]) * 0.5, b[0], b[1]);
  const g = Object.assign({}, genome);
  g.rootPitch = jit(g.rootPitch, B.rootPitch);
  g.hSlope = g.hSlope.map((v) => jit(v, B.hSlope));
  g.vSlope = g.vSlope.map((v) => jit(v, B.vSlope));
  g.pitchChange = g.pitchChange.map((v, i) => (i === 0 ? 0.0 : jit(v, B.pitchChange)));
  g.scaleDown = g.scaleDown.map((v, i) => (i === 0 ? 0.0 : jit(v, B.scaleDown)));
  g.wLen = jit(g.wLen, B.wLen); g.wRad = jit(g.wRad, B.wRad); g.wRot = jit(g.wRot, B.wRot);
  g.wArc = jit(g.wArc, B.wArc); g.wHRate = jit(g.wHRate, B.wHRate); g.wSRate = jit(g.wSRate, B.wSRate);
  g.wFlatTip = jit(g.wFlatTip, B.wFlatTip); g.wPointyX = jit(g.wPointyX, B.wPointyX);
  return g;
}

// ---------------------------------------------------------------- decode
function buildSections(genome) {
  const sections = [];
  const ds = TOTAL_SPAN_M / (N - 1);
  let h = 0.0, vv = 0.0, pitch = genome.rootPitch, scale = 1.0;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    if (i > 0) {
      h += 0.5 * (genome.hSlope[i - 1] + genome.hSlope[i]) * ds;
      vv += 0.5 * (genome.vSlope[i - 1] + genome.vSlope[i]) * ds;
      pitch = clamp(pitch + genome.pitchChange[i], genome.rootPitch - TWIST_LIMIT, genome.rootPitch + TWIST_LIMIT);
      scale = Math.max(SCALE_MIN, scale - genome.scaleDown[i]);
    }
    sections.push({
      airfoil_type: genome.airfoil,
      spanwise_position: TOTAL_SPAN_M * t,
      section_scale: scale,
      chord_length: ROOT_CHORD_M * scale,
      root_pitch_angle_deg: pitch,
      horizontal_displacement: i === 0 ? 0.0 : h,
      vertical_displacement: i === 0 ? 0.0 : vv,
    });
  }
  return sections;
}

function wingtipStations(genome, sections) {
  const arcAbs = Math.abs(genome.wArc) * DEG;
  const theta = genome.wRot * DEG;
  const isPointy = genome.isPointy === true;
  const f = sections[sections.length - 1];
  const p = sections[sections.length - 2];
  const ds = Math.max(f.spanwise_position - p.spanwise_position, 1e-6);
  const sweepSlope = (f.horizontal_displacement - p.horizontal_displacement) / ds;
  const dihedralSlope = (f.vertical_displacement - p.vertical_displacement) / ds;
  const baseThicknessM = profileThickness(AIRFOILS[f.airfoil_type]) * f.chord_length;

  const stations = [];
  for (let i = 1; i <= WINGTIP_SECTIONS; i++) {
    const t = i / WINGTIP_SECTIONS;
    const phi = arcAbs * t;
    const oneMinusCos = 1.0 - Math.cos(phi);
    const spanOffsetM = genome.wLen * Math.sin(phi);
    const bend = genome.wRad * oneMinusCos;
    const verticalBend = Math.cos(theta) * bend;
    const horizontalBend = Math.sin(theta) * bend;
    const horizontalRateBend = genome.wHRate * genome.wLen * oneMinusCos;
    const apex = isPointy ? genome.wPointyX * (genome.wLen * Math.sin(arcAbs)) * smoothstep(t) : 0.0;

    const chordFactor = Math.pow(1.0 - genome.wSRate, i);
    let chordLengthM;
    if (isPointy) {
      const rawEnd = Math.pow(1.0 - genome.wSRate, WINGTIP_SECTIONS);
      const norm = (chordFactor - rawEnd) / Math.max(1e-6, 1.0 - rawEnd);
      chordLengthM = MIN_TIP_CHORD_M + (f.chord_length - MIN_TIP_CHORD_M) * norm;
    } else {
      chordLengthM = Math.max(genome.wFlatTip, f.chord_length * chordFactor);
    }
    const targetThicknessM = Math.max(MIN_TIP_THICKNESS_M, baseThicknessM + (TIP_THICKNESS_M - baseThicknessM) * smoothstep(t));

    stations.push({
      airfoil_type: f.airfoil_type,
      spanwise_position: f.spanwise_position + spanOffsetM,
      section_scale: chordLengthM / ROOT_CHORD_M,
      chord_length: chordLengthM,
      target_thickness_m: targetThicknessM,
      root_pitch_angle_deg: f.root_pitch_angle_deg,
      horizontal_displacement: f.horizontal_displacement + sweepSlope * spanOffsetM + horizontalRateBend + horizontalBend + apex,
      vertical_displacement: f.vertical_displacement + dihedralSlope * spanOffsetM + verticalBend,
    });
  }
  return stations;
}

function airfoilPoints(section) {
  const coords = AIRFOILS[section.airfoil_type];
  const zScale = section.target_thickness_m
    ? (section.target_thickness_m * MM) / profileThickness(coords)
    : section.chord_length * MM;
  const points = [];
  for (let i = 0; i < coords.length; i++) {
    const raw = coords[i];
    const r = rotate2d(raw[0] * section.chord_length * MM, raw[1] * zScale, section.root_pitch_angle_deg);
    points.push([r[0] + section.horizontal_displacement * MM, r[1] + section.vertical_displacement * MM]);
  }
  return points;
}
function airfoilSketch(section) { return polygon(airfoilPoints(section)); }

// ---------------------------------------------------------------- assemble
const parentA = buildGenome(Math.floor(seedA));
const parentB = buildGenome(Math.floor(seedB));
let genome = crossover(parentA, parentB, Math.floor(xoSeed));
genome = mutate(genome, mutAmt, Math.floor(xoSeed) + 12345);
if (showWho === "Parent A") genome = parentA;
else if (showWho === "Parent B") genome = parentB;

const sections = buildSections(genome);
const profiles = [];
const heights = [];
for (const s of sections) { profiles.push(airfoilSketch(s)); heights.push(-s.spanwise_position * MM); }
for (const st of wingtipStations(genome, sections)) { profiles.push(airfoilSketch(st)); heights.push(-st.spanwise_position * MM); }

const lofted = loft(profiles, heights, { edgeLength: 20.0, boundsPadding: 35.0 });
const wing = lofted.rotateX(90).color(genome.isPointy ? "#D67F4F" : "#4F8FD6");

verify.notEmpty("generated wing solid", wing);
verify.greaterThan("wing volume", wing.volume(), 0);

return { Wing: wing };
""".lstrip()


def _common_replacements(config: GeneratorConfig) -> dict[str, str]:
    """Token → value map shared by every interactive template (all from config)."""
    g = config.genes
    airfoils = normalized_airfoil_population(
        tuple(config.allowed_airfoils), config.airfoil_points_per_profile
    )
    names = list(config.allowed_airfoils)

    return {
        "__AIRFOILS_JSON__": json.dumps(airfoils),
        "__AIRFOIL_NAMES_JSON__": json.dumps(names),
        "__DEFAULT_AIRFOIL__": json.dumps(names[0]),
        "__NUM_SECTIONS__": str(config.num_sections),
        "__TOTAL_SPAN__": repr(config.total_span_m),
        "__ROOT_CHORD__": repr(config.root_chord_m),
        "__SCALE_MIN__": repr(config.scale_min),
        "__TWIST_MAX__": repr(config.validation.max_total_pitch_twist_deg),
        "__WINGTIP_SECTIONS__": str(config.wingtip_sections),
        "__MIN_TIP_CHORD_M__": repr(config.winglet_min_tip_chord_m),
        "__MIN_TIP_THICKNESS_M__": repr(config.winglet_min_tip_thickness_m),
        "__TIP_THICKNESS_M__": repr(config.winglet_tip_thickness_m),
        "__MAX_DSLOPE__": repr(config.validation.max_slope_change_per_section),
        "__POINTY_PROB__": repr(config.gene_space.winglet_pointy_probability),
        "__RP_MIN__": repr(g.root_pitch_deg.minimum),
        "__RP_MAX__": repr(g.root_pitch_deg.maximum),
        "__HS_MIN__": repr(g.horizontal_slope.minimum),
        "__HS_MAX__": repr(g.horizontal_slope.maximum),
        "__VS_MIN__": repr(g.vertical_slope.minimum),
        "__VS_MAX__": repr(g.vertical_slope.maximum),
        "__PC_MIN__": repr(g.pitch_change_deg.minimum),
        "__PC_MAX__": repr(g.pitch_change_deg.maximum),
        "__SD_MIN__": repr(g.scale_down.minimum),
        "__SD_MAX__": repr(g.scale_down.maximum),
        "__WLEN_MIN__": repr(g.winglet_ellipse_length.minimum),
        "__WLEN_MAX__": repr(g.winglet_ellipse_length.maximum),
        "__WRAD_MIN__": repr(g.winglet_ellipse_radius.minimum),
        "__WRAD_MAX__": repr(g.winglet_ellipse_radius.maximum),
        "__WROT_MIN__": repr(g.winglet_ellipse_rotation_deg.minimum),
        "__WROT_MAX__": repr(g.winglet_ellipse_rotation_deg.maximum),
        "__WARC_MIN__": repr(g.winglet_arc_angle_deg.minimum),
        "__WARC_MAX__": repr(g.winglet_arc_angle_deg.maximum),
        "__WHR_MIN__": repr(g.winglet_horizontal_rate.minimum),
        "__WHR_MAX__": repr(g.winglet_horizontal_rate.maximum),
        "__WSR_MIN__": repr(g.winglet_scale_rate.minimum),
        "__WSR_MAX__": repr(g.winglet_scale_rate.maximum),
        "__WFTC_MIN__": repr(g.winglet_flat_tip_chord_m.minimum),
        "__WFTC_MAX__": repr(g.winglet_flat_tip_chord_m.maximum),
        "__WPTX_MIN__": repr(g.winglet_pointy_tip_x_frac.minimum),
        "__WPTX_MAX__": repr(g.winglet_pointy_tip_x_frac.maximum),
    }


def _render(template: str, config: GeneratorConfig) -> str:
    source = template
    for token, value in _common_replacements(config).items():
        source = source.replace(token, value)
    return source


def build_slider_design(config: GeneratorConfig) -> str:
    """Genetics-lab design: seed-driven random genomes + crossover/mutation."""
    return _render(_TEMPLATE, config)


def build_manual_design(config: GeneratorConfig) -> str:
    """Fully manual design: every gene is a single slider you tune yourself."""
    return _render(_MANUAL_TEMPLATE, config)


_MANUAL_TEMPLATE = r"""
// =====================================================================
// MANUAL wing + winglet — every gene is a single slider you tune yourself.
// Per-section sweep/dihedral are driven by 3-point SLOPE profiles (root / mid /
// tip); the mid point lets you create an inflection. All ranges = config bounds.
// Frame: X = chord, Y = span, Z = vertical up. Mirrors genome.py geometry.
// =====================================================================

const AIRFOILS = __AIRFOILS_JSON__;
const AIRFOIL_NAMES = __AIRFOIL_NAMES_JSON__;

const MM = 1000.0;
const DEG = Math.PI / 180.0;

const N = __NUM_SECTIONS__;
const TOTAL_SPAN_M = __TOTAL_SPAN__;
const ROOT_CHORD_M = __ROOT_CHORD__;
const SCALE_MIN = __SCALE_MIN__;
const WINGTIP_SECTIONS = __WINGTIP_SECTIONS__;
const MIN_TIP_CHORD_M = __MIN_TIP_CHORD_M__;
const MIN_TIP_THICKNESS_M = __MIN_TIP_THICKNESS_M__;
const TIP_THICKNESS_M = __TIP_THICKNESS_M__;

// ---------------------------------------------------------------- controls
const airfoilName = Param.choice("Airfoil profile", __DEFAULT_AIRFOIL__, AIRFOIL_NAMES);
const rootPitch   = Param.number("Root pitch", 1.5, { min: __RP_MIN__, max: __RP_MAX__, unit: "deg" });
const twistDeg    = Param.number("Twist (washout)", 0.0, { min: -__TWIST_MAX__, max: __TWIST_MAX__, unit: "deg" });
const tipScale    = Param.number("Tip taper ratio", 0.7, { min: SCALE_MIN, max: 1.0 });

const hsRoot = Param.number("Sweep slope @root", 0.05, { min: __HS_MIN__, max: __HS_MAX__ });
const hsMid  = Param.number("Sweep slope @mid", 0.08, { min: __HS_MIN__, max: __HS_MAX__ });
const hsTip  = Param.number("Sweep slope @tip", 0.10, { min: __HS_MIN__, max: __HS_MAX__ });
const vsRoot = Param.number("Dihedral slope @root", 0.0, { min: __VS_MIN__, max: __VS_MAX__ });
const vsMid  = Param.number("Dihedral slope @mid", 0.10, { min: __VS_MIN__, max: __VS_MAX__ });
const vsTip  = Param.number("Dihedral slope @tip", 0.20, { min: __VS_MIN__, max: __VS_MAX__ });

const wgPointy  = Param.bool("Winglet pointy tip", false);
const wgLength  = Param.number("Winglet ellipse length", 0.25, { min: __WLEN_MIN__, max: __WLEN_MAX__, unit: "m" });
const wgRadius  = Param.number("Winglet ellipse radius", 0.2, { min: __WRAD_MIN__, max: __WRAD_MAX__, unit: "m" });
const wgRot     = Param.number("Winglet ellipse rotation", 20.0, { min: __WROT_MIN__, max: __WROT_MAX__, unit: "deg" });
const wgArc     = Param.number("Winglet arc angle", 60.0, { min: __WARC_MIN__, max: __WARC_MAX__, unit: "deg" });
const wgHRate   = Param.number("Winglet sweep rate", 0.1, { min: __WHR_MIN__, max: __WHR_MAX__ });
const wgSRate   = Param.number("Winglet scale-down rate", 0.1, { min: __WSR_MIN__, max: __WSR_MAX__ });
const wgFlatTip = Param.number("Flat tip chord (if flat)", 0.2, { min: __WFTC_MIN__, max: __WFTC_MAX__, unit: "m" });
const wgPointyX = Param.number("Pointy apex lean (frac of reach)", 0.0, { min: __WPTX_MIN__, max: __WPTX_MAX__ });

// ---------------------------------------------------------------- helpers
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function smoothstep(t) { return 3.0 * t * t - 2.0 * t * t * t; }
function rotate2d(x, z, deg) { const a = deg * DEG, c = Math.cos(a), s = Math.sin(a); return [x * c - z * s, x * s + z * c]; }
function profileThickness(coords) {
  let mn = Infinity, mx = -Infinity;
  for (const p of coords) { mn = Math.min(mn, p[1]); mx = Math.max(mx, p[1]); }
  return Math.max(mx - mn, 1e-9);
}
function hSlopeAt(t) { return t < 0.5 ? lerp(hsRoot, hsMid, t / 0.5) : lerp(hsMid, hsTip, (t - 0.5) / 0.5); }
function vSlopeAt(t) { return t < 0.5 ? lerp(vsRoot, vsMid, t / 0.5) : lerp(vsMid, vsTip, (t - 0.5) / 0.5); }

function buildSections() {
  const sections = [];
  const ds = TOTAL_SPAN_M / (N - 1);
  let h = 0.0, vv = 0.0, prevHs = hSlopeAt(0), prevVs = vSlopeAt(0);
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const hs = hSlopeAt(t), vs = vSlopeAt(t);
    if (i > 0) { h += 0.5 * (prevHs + hs) * ds; vv += 0.5 * (prevVs + vs) * ds; }
    prevHs = hs; prevVs = vs;
    const scale = clamp(1.0 + (tipScale - 1.0) * t, SCALE_MIN, 1.0);
    sections.push({
      airfoil_type: airfoilName,
      spanwise_position: TOTAL_SPAN_M * t,
      section_scale: scale,
      chord_length: ROOT_CHORD_M * scale,
      root_pitch_angle_deg: rootPitch + twistDeg * t,
      horizontal_displacement: i === 0 ? 0.0 : h,
      vertical_displacement: i === 0 ? 0.0 : vv,
    });
  }
  return sections;
}

function wingtipStations(sections) {
  const arcAbs = Math.abs(wgArc) * DEG;
  const theta = wgRot * DEG;
  const isPointy = wgPointy === true;
  const f = sections[sections.length - 1];
  const p = sections[sections.length - 2];
  const ds = Math.max(f.spanwise_position - p.spanwise_position, 1e-6);
  const sweepSlope = (f.horizontal_displacement - p.horizontal_displacement) / ds;
  const dihedralSlope = (f.vertical_displacement - p.vertical_displacement) / ds;
  const baseThicknessM = profileThickness(AIRFOILS[f.airfoil_type]) * f.chord_length;

  const stations = [];
  for (let i = 1; i <= WINGTIP_SECTIONS; i++) {
    const t = i / WINGTIP_SECTIONS;
    const phi = arcAbs * t;
    const oneMinusCos = 1.0 - Math.cos(phi);
    const spanOffsetM = wgLength * Math.sin(phi);
    const bend = wgRadius * oneMinusCos;
    const verticalBend = Math.cos(theta) * bend;
    const horizontalBend = Math.sin(theta) * bend;
    const horizontalRateBend = wgHRate * wgLength * oneMinusCos;
    const apex = isPointy ? wgPointyX * (wgLength * Math.sin(arcAbs)) * smoothstep(t) : 0.0;

    const chordFactor = Math.pow(1.0 - wgSRate, i);
    let chordLengthM;
    if (isPointy) {
      const rawEnd = Math.pow(1.0 - wgSRate, WINGTIP_SECTIONS);
      const norm = (chordFactor - rawEnd) / Math.max(1e-6, 1.0 - rawEnd);
      chordLengthM = MIN_TIP_CHORD_M + (f.chord_length - MIN_TIP_CHORD_M) * norm;
    } else {
      chordLengthM = Math.max(wgFlatTip, f.chord_length * chordFactor);
    }
    const targetThicknessM = Math.max(MIN_TIP_THICKNESS_M, baseThicknessM + (TIP_THICKNESS_M - baseThicknessM) * smoothstep(t));

    stations.push({
      airfoil_type: f.airfoil_type,
      spanwise_position: f.spanwise_position + spanOffsetM,
      section_scale: chordLengthM / ROOT_CHORD_M,
      chord_length: chordLengthM,
      target_thickness_m: targetThicknessM,
      root_pitch_angle_deg: f.root_pitch_angle_deg,
      horizontal_displacement: f.horizontal_displacement + sweepSlope * spanOffsetM + horizontalRateBend + horizontalBend + apex,
      vertical_displacement: f.vertical_displacement + dihedralSlope * spanOffsetM + verticalBend,
    });
  }
  return stations;
}

function airfoilPoints(section) {
  const coords = AIRFOILS[section.airfoil_type];
  const zScale = section.target_thickness_m
    ? (section.target_thickness_m * MM) / profileThickness(coords)
    : section.chord_length * MM;
  const points = [];
  for (let i = 0; i < coords.length; i++) {
    const raw = coords[i];
    const r = rotate2d(raw[0] * section.chord_length * MM, raw[1] * zScale, section.root_pitch_angle_deg);
    points.push([r[0] + section.horizontal_displacement * MM, r[1] + section.vertical_displacement * MM]);
  }
  return points;
}
function airfoilSketch(section) { return polygon(airfoilPoints(section)); }

const sections = buildSections();
const profiles = [];
const heights = [];
for (const s of sections) { profiles.push(airfoilSketch(s)); heights.push(-s.spanwise_position * MM); }
for (const st of wingtipStations(sections)) { profiles.push(airfoilSketch(st)); heights.push(-st.spanwise_position * MM); }

const lofted = loft(profiles, heights, { edgeLength: 20.0, boundsPadding: 35.0 });
const wing = lofted.rotateX(90).color(wgPointy ? "#D67F4F" : "#4F8FD6");

verify.notEmpty("generated wing solid", wing);
verify.greaterThan("wing volume", wing.volume(), 0);

return { Wing: wing };
""".lstrip()
