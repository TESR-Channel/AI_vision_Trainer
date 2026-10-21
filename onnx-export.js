/* =====================================================================
   TESR AI Vision Web Trainer - ONNX exporter (runs 100% in the browser)

   Turns the trained web model into ONE standard .onnx file:
     MobileNetV2 backbone (the exact weights used in the browser)
     + your classifier head (+ box head for detection)
   No Python, no TensorFlow, no server: the weights already live in this
   page, so we write the ONNX protobuf by hand.

   Model contract (opset 13, static batch 1 - TensorRT friendly):
     input  images  float32 [1,3,224,224]  RGB, 0..1 (like YOLO)
   classification:
     output probs   [1,nc]  softmax                 (output #0)
            feat    [1,1280] backbone embedding      (similarity gate)
   detection:
     output output0 [1,4+nc,N]  YOLO-style candidates (output #0)
                    cx,cy,w,h in 224-pixel space + per-class scores,
                    background classes zeroed, similarity gate applied.
                    Same layout as an Ultralytics export -> TESR ROI Studio
                    and any YOLO decoder (conf filter + NMS) can read it.
            probs   [1,nc]  whole-frame class probabilities
            box     [1,4]   whole-frame fast box (normalized cx,cy,w,h)
            feat    [1,1280]

   Works in the page (window.TESR_ONNX) and in Node (module.exports),
   which is how it is tested against TensorFlow.js + onnxruntime.
   ===================================================================== */
(function (root) {
  'use strict';

  /* ---------------- tiny protobuf writer ---------------- */
  function PB() { this.parts = []; this.len = 0; }
  PB.prototype.raw = function (u8) { this.parts.push(u8); this.len += u8.length; };
  PB.prototype.varint = function (v) {
    let n = BigInt(v);
    if (n < 0n) n += 1n << 64n;                       // int64 two's complement
    const out = [];
    while (n >= 0x80n) { out.push(Number(n & 0x7fn) | 0x80); n >>= 7n; }
    out.push(Number(n));
    this.raw(Uint8Array.from(out));
  };
  PB.prototype.tag = function (field, wire) { this.varint((field << 3) | wire); };
  PB.prototype.int = function (field, v) { this.tag(field, 0); this.varint(v); };
  PB.prototype.bytes = function (field, u8) { this.tag(field, 2); this.varint(u8.length); this.raw(u8); };
  PB.prototype.str = function (field, s) { this.bytes(field, new TextEncoder().encode(s)); };
  PB.prototype.msg = function (field, pb) {
    this.tag(field, 2); this.varint(pb.len);
    for (const p of pb.parts) this.raw(p);
  };
  PB.prototype.f32 = function (field, v) {
    this.tag(field, 5);
    const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this.raw(b);
  };
  PB.prototype.finish = function () {
    const out = new Uint8Array(this.len); let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    return out;
  };

  /* ---------------- ONNX graph builder ---------------- */
  const FLOAT = 1, INT64 = 7;
  function Graph() { this.nodes = []; this.inits = []; this.n = 0; this.names = new Set(); }
  Graph.prototype.uniq = function (base) {
    let s = String(base).replace(/[^A-Za-z0-9_]/g, '_');
    if (!this.names.has(s)) { this.names.add(s); return s; }
    let k = 1; while (this.names.has(s + '_' + k)) k++;
    s = s + '_' + k; this.names.add(s); return s;
  };
  Graph.prototype.init = function (base, dims, data, type) {
    const name = this.uniq(base);
    this.inits.push({name, dims, data, type: type || FLOAT});
    return name;
  };
  Graph.prototype.f = function (base, dims, arr) { return this.init(base, dims, Float32Array.from(arr)); };
  Graph.prototype.i64 = function (base, arr) {
    return this.init(base, [arr.length], BigInt64Array.from(arr.map(v => BigInt(v))), INT64);
  };
  Graph.prototype.op = function (type, inputs, attrs, outBase, nOut) {
    const outs = [];
    for (let i = 0; i < (nOut || 1); i++) outs.push(this.uniq((outBase || type.toLowerCase()) + (i ? '_' + i : '')));
    this.nodes.push({type, inputs, outputs: outs, attrs: attrs || {}, name: this.uniq('n' + (this.n++) + '_' + type)});
    return nOut ? outs : outs[0];
  };
  /* rename an op's output to a fixed graph output name */
  Graph.prototype.out = function (type, inputs, attrs, name) {
    this.names.add(name);
    this.nodes.push({type, inputs, outputs: [name], attrs: attrs || {}, name: this.uniq('n' + (this.n++) + '_' + type)});
    return name;
  };

  function attrPB(name, v) {
    const a = new PB(); a.str(1, name);
    if (v && v.ints) { for (const x of v.ints) a.int(8, x); a.int(20, 7); }
    else if (v && v.floats) { for (const x of v.floats) a.f32(7, x); a.int(20, 6); }
    else if (v && v.f !== undefined) { a.f32(2, v.f); a.int(20, 1); }
    else if (v && v.s !== undefined) { a.str(4, v.s); a.int(20, 3); }
    else { a.int(3, v.i !== undefined ? v.i : v); a.int(20, 2); }
    return a;
  }
  function valueInfo(name, dims) {
    const shape = new PB();
    for (const d of dims) { const dim = new PB(); dim.int(1, d); shape.msg(1, dim); }
    const tt = new PB(); tt.int(1, FLOAT); tt.msg(2, shape);
    const tp = new PB(); tp.msg(1, tt);
    const vi = new PB(); vi.str(1, name); vi.msg(2, tp);
    return vi;
  }
  Graph.prototype.serialize = function (input, outputs, meta) {
    const g = new PB();
    for (const nd of this.nodes) {
      const p = new PB();
      for (const i of nd.inputs) p.str(1, i);
      for (const o of nd.outputs) p.str(2, o);
      p.str(3, nd.name); p.str(4, nd.type);
      for (const k of Object.keys(nd.attrs)) p.msg(5, attrPB(k, nd.attrs[k]));
      g.msg(1, p);
    }
    g.str(2, 'tesr_web_model');
    for (const t of this.inits) {
      const p = new PB();
      for (const d of t.dims) p.int(1, d);
      p.int(2, t.type); p.str(8, t.name);
      p.bytes(9, new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
      g.msg(5, p);
    }
    g.msg(11, valueInfo(input.name, input.dims));
    for (const o of outputs) g.msg(12, valueInfo(o.name, o.dims));
    const m = new PB();
    m.int(1, 7);                                   // IR v7 (onnx 1.8+, TensorRT 8+)
    m.str(2, 'TESR AI Vision Web Trainer');
    m.str(3, '1.0');
    m.msg(7, g);
    const ops = new PB(); ops.str(1, ''); ops.int(2, 13); m.msg(8, ops);
    for (const k of Object.keys(meta || {})) {
      const e = new PB(); e.str(1, k); e.str(2, String(meta[k])); m.msg(14, e);
    }
    return m.finish();
  };

  /* ---------------- TensorFlow.js graph -> ONNX (NCHW) ---------------- */
  function clean(n) { return n.replace(/^\^/, '').replace(/:0$/, ''); }
  function samePads(inp, k, s, d) {
    const out = Math.ceil(inp / s);
    const total = Math.max((out - 1) * s + (k - 1) * d + 1 - inp, 0);
    return [Math.floor(total / 2), total - Math.floor(total / 2)];
  }
  function attrVal(node, name, dflt) {
    const p = node.attrParams && node.attrParams[name];
    return (p && p.value !== undefined) ? p.value : dflt;
  }
  function firstGraphNode(model, pred) {
    const nodes = model.executor.graph.nodes;
    for (const k of Object.keys(nodes)) if (pred(nodes[k])) return nodes[k];
    return null;
  }
  /* the input of the embedding pool = the 7x7 feature map we need */
  function findFeatureNode(model) {
    const pool = firstGraphNode(model, n => /Logits\/AvgPool$/.test(n.name)) ||
                 firstGraphNode(model, n => n.op === 'AvgPool') ||
                 firstGraphNode(model, n => n.op === 'Mean');
    if (!pool) throw new Error('could not find the embedding pool in the backbone graph');
    return {pool: pool.name, fmap: clean(pool.inputNames[0])};
  }

  /* Emits the backbone into g, from `inputName` ([1,3,S,S], already scaled
     the way the TF.js graph expects) up to the tensor `outNode`.
     Returns {name, shape:[C,H,W]} of that tensor in NCHW. */
  function convertBackbone(g, model, inputName, size, outNode) {
    const nodes = model.executor.graph.nodes;
    const weights = model.weights;
    const done = {};           // tfName -> {name, shape} | {const: Float32Array, shape}

    function constOf(tfName) {
      const n = clean(tfName);
      if (weights[n]) {
        const t = weights[n][0];
        return {const: t.dataSync(), shape: t.shape.slice()};
      }
      const nd = nodes[n];
      if (nd && nd.op === 'Identity') return constOf(nd.inputNames[0]);
      return null;
    }
    function visit(tfName) {
      const n = clean(tfName);
      if (done[n]) return done[n];
      const c = constOf(n);
      if (c) return (done[n] = c);
      const nd = nodes[n];
      if (!nd) throw new Error('graph node not found: ' + n);
      const r = convert(nd);
      done[n] = r;
      return r;
    }
    function toChan(c, C) {                // per-channel const -> [C,1,1] initializer
      const v = c.const;
      if (v.length === 1) return g.f('k', [], [v[0]]);
      if (v.length !== C) throw new Error('constant of length ' + v.length + ' cannot broadcast to ' + C + ' channels');
      return g.f('k', [C, 1, 1], v);
    }
    function activation(x, act, alpha) {
      act = String(act || '').toLowerCase();
      if (!act || act === 'linear') return x;
      if (act === 'relu') return g.op('Relu', [x]);
      if (act === 'relu6') return g.op('Clip', [x, g.f('zero', [], [0]), g.f('six', [], [6])]);
      if (act === 'leakyrelu') return g.op('LeakyRelu', [x], {alpha: {f: alpha || 0.2}});
      if (act === 'elu') return g.op('Elu', [x]);
      if (act === 'sigmoid') return g.op('Sigmoid', [x]);
      throw new Error('unsupported fused activation: ' + act);
    }
    function conv(nd, depthwise) {
      const x = visit(nd.inputNames[0]);
      const w = visit(nd.inputNames[1]);
      if (!w.const) throw new Error('conv filter is not constant: ' + nd.name);
      const [kh, kw, ci, co] = w.shape;     // HWIO (depthwise: H,W,C,mult)
      const strides = attrVal(nd, 'strides', [1, 1, 1, 1]);
      const dil = attrVal(nd, 'dilations', [1, 1, 1, 1]);
      const fmt = String(attrVal(nd, 'dataFormat', 'NHWC')).toUpperCase();
      if (fmt !== 'NHWC') throw new Error('only NHWC graphs are supported (' + nd.name + ')');
      const sh = strides[1], sw = strides[2], dh = dil[1], dw = dil[2];
      const [C, H, W] = x.shape;
      let pt = 0, pb = 0, pl = 0, pr = 0;
      const pad = attrVal(nd, 'pad', 'valid');
      if (pad === 'same') { [pt, pb] = samePads(H, kh, sh, dh); [pl, pr] = samePads(W, kw, sw, dw); }
      else if (pad === 'explicit') {
        const e = attrVal(nd, 'explicitPaddings', []);
        pt = e[2]; pb = e[3]; pl = e[4]; pr = e[5];
      }
      const outC = depthwise ? ci * co : co;
      const Wt = new Float32Array(outC * (depthwise ? 1 : ci) * kh * kw);
      const src = w.const;
      if (depthwise) {                       // [kh,kw,C,M] -> [C*M,1,kh,kw]
        for (let y = 0; y < kh; y++) for (let xx = 0; xx < kw; xx++)
          for (let c = 0; c < ci; c++) for (let m = 0; m < co; m++)
            Wt[((c * co + m) * kh + y) * kw + xx] = src[((y * kw + xx) * ci + c) * co + m];
      } else {                               // [kh,kw,I,O] -> [O,I,kh,kw]
        for (let y = 0; y < kh; y++) for (let xx = 0; xx < kw; xx++)
          for (let i = 0; i < ci; i++) for (let o = 0; o < co; o++)
            Wt[((o * ci + i) * kh + y) * kw + xx] = src[((y * kw + xx) * ci + i) * co + o];
      }
      const ins = [x.name, g.init('w', [outC, depthwise ? 1 : ci, kh, kw], Wt)];
      const fused = attrVal(nd, 'fusedOps', []) || [];
      const hasBias = String(fused[0] || '').toLowerCase() === 'biasadd';
      if (fused.length && String(fused[0]).toLowerCase() === 'fusedbatchnorm')
        throw new Error('fused batch norm is not supported (' + nd.name + ')');
      if (hasBias) {
        const b = visit(nd.inputNames[2]);
        ins.push(g.f('b', [outC], b.const));
      }
      let y = g.op('Conv', ins, {
        kernel_shape: {ints: [kh, kw]}, strides: {ints: [sh, sw]},
        dilations: {ints: [dh, dw]}, pads: {ints: [pt, pl, pb, pr]},
        group: depthwise ? ci : 1});
      if (fused.length > 1) y = activation(y, fused[1], attrVal(nd, 'leakyreluAlpha', 0.2));
      const oh = Math.floor((H + pt + pb - ((kh - 1) * dh + 1)) / sh) + 1;
      const ow = Math.floor((W + pl + pr - ((kw - 1) * dw + 1)) / sw) + 1;
      return {name: y, shape: [outC, oh, ow]};
    }
    function binary(nd, onnxOp) {
      const a = visit(nd.inputNames[0]), b = visit(nd.inputNames[1]);
      if (a.const && b.const) throw new Error('constant folding not supported: ' + nd.name);
      if (a.const || b.const) {
        const t = a.const ? b : a, c = a.const ? a : b;
        const k = toChan(c, t.shape[0]);
        const ins = a.const ? [k, t.name] : [t.name, k];
        return {name: g.op(onnxOp, ins), shape: t.shape};
      }
      return {name: g.op(onnxOp, [a.name, b.name]), shape: a.shape};
    }
    function convert(nd) {
      switch (nd.op) {
        case 'Placeholder': return {name: inputName, shape: [3, size, size]};
        case 'Identity': case 'StopGradient': case 'Snapshot': return visit(nd.inputNames[0]);
        case 'Conv2D': case '_FusedConv2D': return conv(nd, false);
        case 'DepthwiseConv2dNative': case 'DepthwiseConv2d':
        case 'FusedDepthwiseConv2dNative': return conv(nd, true);
        case 'BiasAdd': case 'Add': case 'AddV2': return binary(nd, 'Add');
        case 'Sub': return binary(nd, 'Sub');
        case 'Mul': return binary(nd, 'Mul');
        case 'AddN': {
          const xs = nd.inputNames.filter(s => !s.startsWith('^')).map(visit);
          return {name: g.op('Sum', xs.map(x => x.name)), shape: xs[0].shape};
        }
        case 'Relu': { const x = visit(nd.inputNames[0]); return {name: activation(x.name, 'relu'), shape: x.shape}; }
        case 'Relu6': { const x = visit(nd.inputNames[0]); return {name: activation(x.name, 'relu6'), shape: x.shape}; }
        case 'FusedBatchNorm': case 'FusedBatchNormV2': case 'FusedBatchNormV3': {
          const x = visit(nd.inputNames[0]);
          const p = [1, 2, 3, 4].map(i => g.f('bn', [x.shape[0]], visit(nd.inputNames[i]).const));
          return {name: g.op('BatchNormalization', [x.name].concat(p),
                    {epsilon: {f: attrVal(nd, 'epsilon', 1e-3)}}), shape: x.shape};
        }
        case 'Pad': case 'PadV2': case 'MirrorPad': {
          const x = visit(nd.inputNames[0]);
          const pv = visit(nd.inputNames[1]).const;         // [[n0,n1],[h0,h1],[w0,w1],[c0,c1]]
          if (pv[0] || pv[1] || pv[6] || pv[7]) throw new Error('padding on batch/channel axes not supported');
          const pads = g.i64('pads', [0, 0, pv[2], pv[4], 0, 0, pv[3], pv[5]]);
          const ins = [x.name, pads];
          if (nd.op === 'PadV2') ins.push(g.f('padv', [], [visit(nd.inputNames[2]).const[0]]));
          const attrs = nd.op === 'MirrorPad'
            ? {mode: {s: String(attrVal(nd, 'mode', 'reflect')).toLowerCase()}} : {};
          return {name: g.op('Pad', ins, attrs),
                  shape: [x.shape[0], x.shape[1] + pv[2] + pv[3], x.shape[2] + pv[4] + pv[5]]};
        }
        default:
          throw new Error('unsupported TensorFlow.js op "' + nd.op + '" at ' + nd.name);
      }
    }
    return visit(outNode);
  }

  /* ---------------- dense heads (tf.Sequential of Dense layers) ---------------- */
  /* grid detector head: Conv2D layers on the [G,G,C] map -> sigmoid -> YOLO-style output0 */
  function emitGridHead(g, fmapName, C, G, seq, nc, S) {
    const convs = seq.layers.filter(l => l.getClassName() === 'Conv2D');
    let x = fmapName, cin = C;
    convs.forEach((l, k) => {
      const [kw, b] = l.getWeights();                        // [kh,kw,in,out]
      const [kh, kk, ci, co] = kw.shape, kd = kw.dataSync();
      if (ci !== cin) throw new Error('grid head shape mismatch');
      const Wt = new Float32Array(co * ci * kh * kk);
      for (let y = 0; y < kh; y++) for (let xx = 0; xx < kk; xx++)
        for (let i = 0; i < ci; i++) for (let o = 0; o < co; o++)
          Wt[((o * ci + i) * kh + y) * kk + xx] = kd[((y * kk + xx) * ci + i) * co + o];
      x = g.op('Conv', [x, g.init('grid_w', [co, ci, kh, kk], Wt), g.f('grid_b', [co], b.dataSync())],
               {kernel_shape: {ints: [kh, kk]}, strides: {ints: [1, 1]},
                pads: {ints: [kh >> 1, kk >> 1, kh >> 1, kk >> 1]}});
      const act = (l.getConfig().activation || 'linear').toLowerCase();
      if (act === 'relu') x = g.op('Relu', [x]);
      else if (act !== 'linear') throw new Error('unsupported grid activation ' + act);
      cin = co;
    });
    if (cin !== nc + 4) throw new Error('grid head must output nc+4 channels');
    const sig = g.op('Sigmoid', [x]);
    const [sc, bx] = g.op('Split', [sig, g.i64('split', [nc, 4])], {axis: 1}, 'grid_part', 2);
    const off = new Float32Array(4 * G * G);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      off[i * G + j] = j * S / G; off[G * G + i * G + j] = i * S / G;
    }
    const boxes = g.op('Add', [g.op('Mul', [bx, g.f('grid_scale', [1, 4, 1, 1], [S / G, S / G, S, S])]),
                               g.f('grid_off', [1, 4, G, G], off)]);
    const out = g.op('Reshape', [g.op('Concat', [boxes, sc], {axis: 1}), g.i64('shape', [1, 4 + nc, G * G])]);
    // strongest cell over all classes -> its box, normalized 0..1
    const flatS = g.op('Reshape', [sc, g.i64('shape', [1, nc, G * G])]);
    const best = g.op('ArgMax', [g.op('ReduceMax', [flatS], {axes: {ints: [1]}, keepdims: 0})], {axis: 1, keepdims: 1});
    const flatB = g.op('Reshape', [boxes, g.i64('shape', [1, 4, G * G])]);
    const picked = g.op('Gather', [flatB, best], {axis: 2});                // [1,4,1,1]
    const topBox = g.op('Mul', [g.op('Reshape', [picked, g.i64('shape', [1, 4])]), g.f('inv_s', [], [1 / S])]);
    return {out, topBox};
  }
  /* spatial box head: Conv2D 1x1 on the [G,G,C] map -> Flatten (NHWC order) -> Dense... */
  function emitSpatialHead(g, fmapName, C, G, seq) {
    const conv = seq.layers.find(l => l.getClassName() === 'Conv2D');
    const [k, b] = conv.getWeights();                      // k: [1,1,C,F]
    const F = k.shape[3], kd = k.dataSync();
    if (k.shape[0] !== 1 || k.shape[1] !== 1 || k.shape[2] !== C) throw new Error('unexpected spatial head ' + k.shape);
    const Wt = new Float32Array(F * C);
    for (let c = 0; c < C; c++) for (let f = 0; f < F; f++) Wt[f * C + c] = kd[c * F + f];
    let y = g.op('Conv', [fmapName, g.init('sbox_w', [F, C, 1, 1], Wt), g.f('sbox_b', [F], b.dataSync())],
                 {kernel_shape: {ints: [1, 1]}, strides: {ints: [1, 1]}, pads: {ints: [0, 0, 0, 0]}});
    const act = (conv.getConfig().activation || 'linear').toLowerCase();
    if (act === 'relu') y = g.op('Relu', [y]);
    else if (act !== 'linear') throw new Error('unsupported spatial activation ' + act);
    y = g.op('Transpose', [y], {perm: {ints: [0, 2, 3, 1]}});               // NCHW -> NHWC like tf.flatten
    y = g.op('Reshape', [y, g.i64('shape', [1, G * G * F])]);
    return emitHead(g, y, denseLayers(seq), 'sigmoid');
  }
  function denseLayers(seq) {
    return seq.layers.filter(l => l.getClassName() === 'Dense').map(l => {
      const [k, b] = l.getWeights();
      return {k: k.dataSync().slice(), kShape: k.shape.slice(), b: b.dataSync().slice(),
              act: (l.getConfig().activation || 'linear').toLowerCase()};
    });
  }
  function emitHead(g, x, layers, finalAct) {
    let y = x;
    layers.forEach((L, i) => {
      // initializers are created once per layer and shared by every use of the head
      if (!L.wName) { L.wName = g.f('dense_w', L.kShape, L.k); L.bName = g.f('dense_b', [L.b.length], L.b); }
      y = g.op('Gemm', [y, L.wName, L.bName]);
      const act = i === layers.length - 1 ? (finalAct || L.act) : L.act;
      if (act === 'relu') y = g.op('Relu', [y]);
      else if (act === 'sigmoid') y = g.op('Sigmoid', [y]);
      else if (act === 'softmax') y = g.op('Softmax', [y], {axis: 1});
      else if (act !== 'linear') throw new Error('unsupported head activation: ' + act);
    });
    return y;
  }

  /* sliding windows over the 7x7 feature grid (kernel sizes in cells) */
  const WINDOW_KERNELS = [7, 5, 3, 2];
  function windowGeometry(G, size) {
    const cell = size / G, geo = [];
    for (const k of WINDOW_KERNELS) {
      if (k > G) continue;
      const n = G - k + 1;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
        geo.push({x0: j * cell, y0: i * cell, s: k * cell, k, n});
    }
    return geo;
  }
  const BG_RE = /^(background|bg|none|empty|nothing)([_\-]|$)/i;
  function isBackground(name) { return BG_RE.test(String(name)); }

  /* ---------------- public: build the full model ---------------- */
  function buildModel(opts) {
    const {net, head, boxHead, boxHeadS, detHead, protos, task} = opts;
    const size = 224;
    const g = new Graph();
    const names = head.classNames.slice();
    const nc = names.length;

    // Reproduce mobilenet.infer() EXACTLY: pixel(0..255) * normalizationConstant + inputMin.
    // TF-Hub MobileNet models use inputRange [0,1] (their graph rescales to [-1,1]
    // internally), so for them this is identity on our 0..1 input. Never hard-code it.
    const nConst = (typeof net.normalizationConstant === 'number') ? net.normalizationConstant : 1 / 255;
    const nMin = (typeof net.inputMin === 'number') ? net.inputMin : 0;
    const scale = nConst * 255;
    let xin = 'images';
    if (Math.abs(scale - 1) > 1e-7) xin = g.op('Mul', [xin, g.f('in_scale', [], [scale])], {}, 'x_scaled');
    if (Math.abs(nMin) > 1e-7) xin = g.op('Add', [xin, g.f('in_min', [], [nMin])], {}, 'x_norm');
    const where = findFeatureNode(net.model);
    const fmap = convertBackbone(g, net.model, xin, size, where.fmap);
    const [C, G, G2] = fmap.shape;
    if (G !== G2) throw new Error('unexpected feature map ' + fmap.shape);

    const feat = g.op('Flatten', [g.op('GlobalAveragePool', [fmap.name])], {axis: 1}, 'gfeat');
    const clsL = denseLayers(head);
    const outputs = [];
    const featOut = g.out('Identity', [feat], {}, 'feat');

    if (task === 'detect' && detHead) {
      // multi-object grid detector: every 7x7 cell -> (cx, cy, w, h) + per-class score
      const det = emitGridHead(g, fmap.name, C, G, detHead, nc, size);   // [1,4+nc,G*G]
      g.out('Identity', [det.out], {}, 'output0');
      outputs.push({name: 'output0', dims: [1, 4 + nc, G * G]});
      g.out('Identity', [emitHead(g, feat, clsL, 'softmax')], {}, 'probs');
      outputs.push({name: 'probs', dims: [1, nc]});
      g.out('Identity', [det.topBox], {}, 'box');                         // box of the strongest cell
      outputs.push({name: 'box', dims: [1, 4]});
    } else if (task === 'detect' && boxHead) {
      const boxL = denseLayers(boxHead);
      const geo = windowGeometry(G, size);
      const N = geo.length;
      // window features: AveragePool per kernel -> [1,C,n*n] -> concat -> [N,C]
      const parts = [];
      for (const k of WINDOW_KERNELS) {
        if (k > G) continue;
        const n = G - k + 1;
        const p = g.op('AveragePool', [fmap.name], {kernel_shape: {ints: [k, k]}, strides: {ints: [1, 1]}});
        parts.push(g.op('Reshape', [p, g.i64('shape', [1, C, n * n])]));
      }
      const cat = parts.length > 1 ? g.op('Concat', parts, {axis: 2}) : parts[0];
      const Wf = g.op('Reshape', [g.op('Transpose', [cat], {perm: {ints: [0, 2, 1]}}), g.i64('shape', [N, C])], {}, 'win_feat');

      const P = emitHead(g, Wf, clsL, 'softmax');              // [N,nc]
      const B = emitHead(g, Wf, boxL, 'sigmoid');              // [N,4] relative to window
      const scale = new Float32Array(N * 4), offs = new Float32Array(N * 4);
      geo.forEach((w, i) => {
        scale.set([w.s, w.s, w.s, w.s], i * 4);
        offs.set([w.x0, w.y0, 0, 0], i * 4);
      });
      const boxes = g.op('Add', [g.op('Mul', [B, g.f('win_scale', [N, 4], scale)]), g.f('win_off', [N, 4], offs)]);

      let S = g.op('Mul', [P, g.f('fg_mask', [1, nc], names.map(n => isBackground(n) ? 0 : 1))]);
      if (protos && protos.C && protos.C.length) {
        const K = protos.C.length;
        const CT = new Float32Array(C * K);
        protos.C.forEach((row, kk) => {
          const nrm = Math.sqrt(row.reduce((a, v) => a + v * v, 0)) + 1e-9;   // unit prototypes
          row.forEach((v, c) => { CT[c * K + kk] = v / nrm; });
        });
        const l2 = g.op('Add', [g.op('ReduceL2', [Wf], {axes: {ints: [1]}, keepdims: 1}), g.f('eps', [], [1e-9])]);
        const sim = g.op('ReduceMax', [g.op('MatMul', [g.op('Div', [Wf, l2]), g.f('protos_t', [C, K], CT)])],
                         {axes: {ints: [1]}, keepdims: 1}, 'win_sim');
        const thr = Math.max(0.2, protos.thr * (opts.windowSimFactor || 0.9));
        const gate = g.op('Cast', [g.op('Greater', [sim, g.f('sim_thr', [], [thr])])], {to: FLOAT});
        S = g.op('Mul', [S, gate]);
      }
      const det = g.op('Concat', [boxes, S], {axis: 1});                     // [N,4+nc]
      const detT = g.op('Transpose', [det], {perm: {ints: [1, 0]}});         // [4+nc,N]
      g.out('Unsqueeze', [detT, g.i64('axes', [0])], {}, 'output0');
      outputs.push({name: 'output0', dims: [1, 4 + nc, N]});

      g.out('Identity', [emitHead(g, feat, clsL, 'softmax')], {}, 'probs');
      outputs.push({name: 'probs', dims: [1, nc]});
      const boxOut = boxHeadS ? emitSpatialHead(g, fmap.name, C, G, boxHeadS)
                              : emitHead(g, feat, boxL, 'sigmoid');
      g.out('Identity', [boxOut], {}, 'box');
      outputs.push({name: 'box', dims: [1, 4]});
    } else {
      g.out('Identity', [emitHead(g, feat, clsL, 'softmax')], {}, 'probs');
      outputs.push({name: 'probs', dims: [1, nc]});
    }
    outputs.push({name: featOut, dims: [1, C]});

    const meta = {
      names: JSON.stringify(Object.fromEntries(names.map((n, i) => [i, n]))),
      task: task === 'detect' ? 'detect' : 'classify',
      imgsz: JSON.stringify([size, size]),
      preprocess: 'RGB, resize (stretch) to 224x224, /255, NCHW',
      input_scale: String(scale), input_min: String(nMin),
      author: 'TESR AI Vision Web Trainer',
    };
    const bytes = g.serialize({name: 'images', dims: [1, 3, size, size]}, outputs, meta);
    return {bytes, outputs: outputs.map(o => o.name), windows: task === 'detect' ? windowGeometry(G, size).length : 0,
            fmapNode: where.fmap, featDim: C, grid: G};
  }

  const api = {buildModel, findFeatureNode, convertBackbone, Graph, windowGeometry,
               WINDOW_KERNELS, isBackground};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TESR_ONNX = api;
})(typeof window !== 'undefined' ? window : this);
