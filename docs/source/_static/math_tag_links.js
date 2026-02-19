(() => {
  let cachedEqnoStyle = null;
  let alignRafId = 0;
  const alignRetryTimeoutIds = [];
  let observerIdleTimeoutId = 0;
  const EQ_ALIGN_SELECTORS = [
    "div.math.eq-align-dep-w",
    "div.math.eq-align-dep-theta",
    "div.math.eq-align-pep",
    "div.math.eq-align-w",
    "div.math.eq-align-theta",
  ];
  const MATH_NODE_SELECTOR = [
    "mjx-container",
    "mjx-tag",
    "mjx-mtext",
    "mjx-mi",
    ".rst-content div.math",
    ".rst-content span.math",
  ].join(", ");
  const ALIGN_RETRY_DELAYS_MS = [120, 320, 800, 1800, 3600, 7000];
  const INLINE_LABEL_TARGETS = {
    C1: "eq-c1",
    C2: "eq-c2",
    C3: "eq-c3",
    C4: "eq-c4",
  };

  function resolveEqnoStyle() {
    if (cachedEqnoStyle) {
      return cachedEqnoStyle;
    }
    const eqnos = document.querySelectorAll(".rst-content div.math > span.eqno");
    for (const eqno of eqnos) {
      const cs = window.getComputedStyle(eqno);
      if (Number.parseFloat(cs.fontSize || "0") > 0) {
        cachedEqnoStyle = {
          fontFamily: cs.fontFamily,
          fontSize: cs.fontSize,
          fontWeight: cs.fontWeight,
          fontStyle: cs.fontStyle,
          lineHeight: cs.lineHeight,
          color: cs.color,
          letterSpacing: cs.letterSpacing,
        };
        return cachedEqnoStyle;
      }
    }
    return null;
  }

  function applyEqnoStyle(element) {
    const s = resolveEqnoStyle();
    if (!s) {
      return;
    }
    element.style.setProperty("font-family", s.fontFamily, "important");
    element.style.setProperty("font-size", s.fontSize, "important");
    element.style.setProperty("font-weight", s.fontWeight, "important");
    element.style.setProperty("font-style", s.fontStyle, "important");
    element.style.setProperty("line-height", s.lineHeight, "important");
    element.style.setProperty("color", s.color, "important");
    element.style.setProperty("letter-spacing", s.letterSpacing, "important");
  }

  function extractLabel(tagText) {
    const label = String(tagText || "")
      .replace(/[()\s]/g, "")
      .trim();
    return /^[A-Za-z0-9-]+$/.test(label) ? label : null;
  }

  function tagToId(tagText) {
    const compact = String(tagText || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "");
    return compact ? `eq-${compact}` : null;
  }

  function bindTag(tagNode) {
    if (!(tagNode instanceof HTMLElement)) {
      return;
    }
    if (tagNode.dataset.anchorBound === "1") {
      return;
    }
    const label = extractLabel(tagNode.textContent);
    if (!label) {
      return;
    }
    const targetId = tagToId(label);
    if (!targetId || !document.getElementById(targetId)) {
      return;
    }

    tagNode.dataset.anchorBound = "1";
    tagNode.classList.add("plain-math-tag");
    tagNode.textContent = "";

    const link = document.createElement("a");
    link.className = "plain-math-tag-link";
    link.href = `#${targetId}`;
    link.textContent = `(${label})`;
    link.setAttribute("aria-label", `Jump to ${label}`);
    applyEqnoStyle(link);

    link.addEventListener("click", (event) => {
      const hash = `#${targetId}`;
      if (window.location.hash === hash) {
        event.preventDefault();
        const target = document.getElementById(targetId);
        if (target) {
          target.scrollIntoView({ block: "center" });
        }
      }
    });

    tagNode.appendChild(link);
  }

  function bindAllTags(root = document) {
    root
      .querySelectorAll('mjx-container[display="true"] mjx-tag')
      .forEach(bindTag);
  }

  function createInlineLink(label, targetId) {
    const link = document.createElement("a");
    link.className = "plain-math-inline-link";
    link.href = `#${targetId}`;
    link.textContent = label;
    link.setAttribute("aria-label", `Jump to ${label}`);

    link.addEventListener("click", (event) => {
      const hash = `#${targetId}`;
      if (window.location.hash === hash) {
        event.preventDefault();
        const target = document.getElementById(targetId);
        if (target) {
          target.scrollIntoView({ block: "center" });
        }
      }
    });
    return link;
  }

  function bindInlineTextNode(node) {
    if (!(node instanceof HTMLElement)) {
      return;
    }
    if (node.dataset.inlineAnchorBound === "1") {
      return;
    }
    if (node.closest("mjx-tag")) {
      return;
    }
    if (node.querySelector("a.plain-math-inline-link")) {
      return;
    }

    const rawText = String(node.textContent || "");
    if (!/\bC[1-4]\b/.test(rawText)) {
      return;
    }
    const parts = rawText.split(/\b(C[1-4])\b/g);
    if (!parts.length) {
      return;
    }

    node.dataset.inlineAnchorBound = "1";
    node.classList.add("plain-math-inline-label");
    node.textContent = "";

    for (const part of parts) {
      const targetId = INLINE_LABEL_TARGETS[part];
      if (targetId && document.getElementById(targetId)) {
        node.appendChild(createInlineLink(part, targetId));
      } else if (part) {
        node.appendChild(document.createTextNode(part));
      }
    }
  }

  function bindSplitInlineLabel(miNode) {
    if (!(miNode instanceof HTMLElement)) {
      return;
    }
    if (miNode.dataset.inlineAnchorBound === "1") {
      return;
    }
    if (miNode.closest("mjx-tag")) {
      return;
    }
    if (String(miNode.textContent || "").trim() !== "C") {
      return;
    }

    const next = miNode.nextElementSibling;
    if (!(next instanceof HTMLElement)) {
      return;
    }
    if (next.dataset.inlineAnchorBound === "1") {
      return;
    }
    if (next.tagName.toLowerCase() !== "mjx-mn") {
      return;
    }

    const digit = String(next.textContent || "").trim();
    const label = `C${digit}`;
    const targetId = INLINE_LABEL_TARGETS[label];
    if (!targetId || !document.getElementById(targetId)) {
      return;
    }

    miNode.dataset.inlineAnchorBound = "1";
    next.dataset.inlineAnchorBound = "1";
    miNode.classList.add("plain-math-inline-label");
    miNode.textContent = "";
    miNode.appendChild(createInlineLink(label, targetId));
    next.textContent = "";
  }

  function bindAllInlineLabels(root = document) {
    root
      .querySelectorAll("mjx-container mjx-mtext")
      .forEach(bindInlineTextNode);
    root
      .querySelectorAll("mjx-container mjx-mi")
      .forEach(bindSplitInlineLabel);
  }

  function clearAlignment(mathBlock, mjx) {
    mathBlock.classList.remove("eq-align-equals-group");
    mjx.style.removeProperty("width");
    mjx.style.removeProperty("padding-left");
    mjx.style.removeProperty("box-sizing");
  }

  function findMainEquals(mjx) {
    const byOperator = Array.from(mjx.querySelectorAll("mjx-mo")).find(
      (node) => String(node.textContent || "").trim() === "="
    );
    if (byOperator) {
      return byOperator;
    }

    // Fallback for render variants that do not expose '=' as mjx-mo.
    return Array.from(mjx.querySelectorAll("*")).find(
      (node) => !node.children.length && String(node.textContent || "").trim() === "="
    );
  }

  function alignEqualsInSelector(selector) {
    const blocks = Array.from(document.querySelectorAll(selector)).filter(
      (node) => node instanceof HTMLElement
    );
    if (!blocks.length) {
      return;
    }

    const measurements = [];
    for (const mathBlock of blocks) {
      const mjx = mathBlock.querySelector('mjx-container[display="true"]');
      if (!(mjx instanceof HTMLElement)) {
        continue;
      }
      clearAlignment(mathBlock, mjx);
      const eqNode = findMainEquals(mjx);
      if (!(eqNode instanceof Element)) {
        continue;
      }
      const mjxRect = mjx.getBoundingClientRect();
      const eqRect = eqNode.getBoundingClientRect();
      const eqX = eqRect.left - mjxRect.left;
      if (!Number.isFinite(eqX) || mjxRect.width <= 0) {
        continue;
      }
      measurements.push({
        mathBlock,
        mjx,
        eqX,
        width: mjxRect.width,
      });
    }

    if (measurements.length < 2) {
      return;
    }

    const targetEqX = Math.max(...measurements.map((m) => m.eqX));
    const groupWidth = Math.max(
      ...measurements.map((m) => m.width + (targetEqX - m.eqX))
    );

    for (const m of measurements) {
      const shift = Math.max(0, targetEqX - m.eqX);
      m.mathBlock.classList.add("eq-align-equals-group");
      m.mjx.style.setProperty("width", `${groupWidth}px`);
      m.mjx.style.setProperty("padding-left", `${shift}px`);
      m.mjx.style.setProperty("box-sizing", "border-box");
    }
  }

  function hasAlignmentTargets(root = document) {
    return EQ_ALIGN_SELECTORS.some((selector) => !!root.querySelector(selector));
  }

  function hasMathContent(root = document) {
    return !!root.querySelector(MATH_NODE_SELECTOR);
  }

  function clearAlignmentRetryTimeouts() {
    while (alignRetryTimeoutIds.length) {
      window.clearTimeout(alignRetryTimeoutIds.pop());
    }
  }

  function scheduleAlignmentRetries() {
    clearAlignmentRetryTimeouts();
    for (const delayMs of ALIGN_RETRY_DELAYS_MS) {
      alignRetryTimeoutIds.push(
        window.setTimeout(() => {
          scheduleEqualsAlignment();
        }, delayMs)
      );
    }
  }

  function armObserverIdleTimeout(observer) {
    if (observerIdleTimeoutId) {
      window.clearTimeout(observerIdleTimeoutId);
    }
    observerIdleTimeoutId = window.setTimeout(() => {
      observer.disconnect();
    }, 20000);
  }

  function nodeHasMathContent(node) {
    if (!(node instanceof Element)) {
      return false;
    }
    if (node.matches(MATH_NODE_SELECTOR)) {
      return true;
    }
    return !!node.querySelector(MATH_NODE_SELECTOR);
  }

  function scheduleEqualsAlignment() {
    if (alignRafId) {
      window.cancelAnimationFrame(alignRafId);
    }
    alignRafId = window.requestAnimationFrame(() => {
      alignRafId = 0;
      EQ_ALIGN_SELECTORS.forEach(alignEqualsInSelector);
    });
  }

  function init() {
    if (!hasMathContent()) {
      return;
    }

    bindAllTags();
    bindAllInlineLabels();
    if (hasAlignmentTargets()) {
      scheduleEqualsAlignment();
      scheduleAlignmentRetries();
    }
    window.addEventListener("resize", scheduleEqualsAlignment, { passive: true });
    window.addEventListener("pageshow", scheduleEqualsAlignment, { passive: true });
    if (document.fonts && document.fonts.ready && typeof document.fonts.ready.then === "function") {
      document.fonts.ready.then(() => {
        scheduleEqualsAlignment();
      });
    }
    const observer = new MutationObserver((mutations) => {
      let sawMathMutation = false;
      let sawAlignmentMutation = false;
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (!nodeHasMathContent(node)) {
            continue;
          }
          sawMathMutation = true;

          if (!(node instanceof Element)) {
            continue;
          }
          if (node.matches("mjx-tag")) {
            bindTag(node);
          }
          if (node.matches("mjx-mtext")) {
            bindInlineTextNode(node);
          }
          if (node.matches("mjx-mi")) {
            bindSplitInlineLabel(node);
          }
          bindAllTags(node);
          bindAllInlineLabels(node);
          if (!sawAlignmentMutation && hasAlignmentTargets(node)) {
            sawAlignmentMutation = true;
          }
        }
      }
      if (!sawMathMutation) {
        return;
      }

      armObserverIdleTimeout(observer);
      if (!sawAlignmentMutation && hasAlignmentTargets()) {
        sawAlignmentMutation = true;
      }
      if (sawAlignmentMutation) {
        scheduleEqualsAlignment();
        scheduleAlignmentRetries();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    armObserverIdleTimeout(observer);
  }

  if (
    window.MathJax &&
    window.MathJax.startup &&
    window.MathJax.startup.promise &&
    typeof window.MathJax.startup.promise.then === "function"
  ) {
    window.MathJax.startup.promise.then(init);
    return;
  }
  window.addEventListener("load", init, { once: true });
})();
