// Runs the pure-JavaScript Rill engine off the main thread, so the page stays responsive while it computes.
importScripts("rill-engine.js");
let engine = null;
onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      engine = new self.RillEngine(new Uint8Array(m.bytes), m.spec);
      postMessage({ id: m.id, ok: true });
    } else if (m.type === "run") {
      const r = engine.run(m.ids, m.seg, m.pos, m.slots);
      postMessage({ id: m.id, ok: true, slot_logits: r.slot_logits, slot_hidden: r.slot_hidden }, [r.slot_logits.buffer, r.slot_hidden.buffer]);
    }
  } catch (err) {
    postMessage({ id: m.id, ok: false, message: String((err && err.message) || err) });
  }
};
