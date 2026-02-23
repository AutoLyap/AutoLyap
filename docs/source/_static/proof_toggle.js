(() => {
  const PROOF_SELECTOR = ".rst-content .proof.docutils.container";

  function normalizeProofLeadParagraph(content) {
    const firstParagraph = content.querySelector(":scope > p");
    if (!(firstParagraph instanceof HTMLParagraphElement)) {
      return;
    }

    const lead = firstParagraph.firstElementChild;
    if (
      lead instanceof HTMLElement &&
      lead.tagName.toLowerCase() === "em" &&
      /^proof\.?$/i.test((lead.textContent || "").trim())
    ) {
      lead.remove();
      const firstNode = firstParagraph.firstChild;
      if (firstNode && firstNode.nodeType === Node.TEXT_NODE) {
        firstNode.textContent = String(firstNode.textContent || "").replace(
          /^\s+/,
          "",
        );
      }
      const paragraphText = String(firstParagraph.textContent || "").trim();
      if (!paragraphText && !firstParagraph.children.length) {
        firstParagraph.remove();
      }
    }
  }

  function updateProofState(proof, open) {
    const content = proof.querySelector(":scope > .proof-content");
    const button = proof.querySelector(":scope > .proof-toggle-header .proof-toggle-button");
    if (!(content instanceof HTMLElement) || !(button instanceof HTMLButtonElement)) {
      return;
    }

    proof.classList.toggle("is-open", open);
    content.hidden = !open;
    button.textContent = open ? "Hide" : "Show";
    button.setAttribute("aria-expanded", String(open));
    button.setAttribute("aria-label", open ? "Hide proof" : "Show proof");
  }

  function initProof(proof) {
    if (!(proof instanceof HTMLElement) || proof.dataset.proofToggleReady === "1") {
      return;
    }

    const content = document.createElement("div");
    content.className = "proof-content";

    while (proof.firstChild) {
      content.appendChild(proof.firstChild);
    }
    normalizeProofLeadParagraph(content);

    const header = document.createElement("div");
    header.className = "proof-toggle-header";

    const title = document.createElement("em");
    title.className = "proof-toggle-title";
    title.textContent = "Proof.";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "proof-toggle-button";
    button.setAttribute("aria-label", "Show proof");

    header.appendChild(title);
    header.appendChild(button);
    proof.appendChild(header);
    proof.appendChild(content);

    proof.dataset.proofToggleReady = "1";
    updateProofState(proof, false);

    button.addEventListener("click", () => {
      const isOpen = proof.classList.contains("is-open");
      updateProofState(proof, !isOpen);
    });
  }

  function initAllProofs() {
    document.querySelectorAll(PROOF_SELECTOR).forEach(initProof);
  }

  function hashTargetElement() {
    const hash = window.location.hash;
    if (!hash || hash.length < 2) {
      return null;
    }

    const id = decodeHashId(hash.slice(1));
    return document.getElementById(id);
  }

  function decodeHashId(id) {
    try {
      return decodeURIComponent(id);
    } catch (_err) {
      return id;
    }
  }

  function revealProofForTarget(target) {
    if (!(target instanceof Element)) {
      return;
    }
    const proof = target.closest(PROOF_SELECTOR);
    if (!(proof instanceof HTMLElement)) {
      return;
    }
    updateProofState(proof, true);
  }

  function revealProofForHashTarget() {
    const target = hashTargetElement();
    revealProofForTarget(target);
  }

  function bindHashAwareLinkExpansion() {
    document.addEventListener("click", (event) => {
      const source = event.target;
      if (!(source instanceof Element)) {
        return;
      }
      const link = source.closest('a[href*="#"]');
      if (!(link instanceof HTMLAnchorElement)) {
        return;
      }

      let url;
      try {
        url = new URL(link.href, window.location.href);
      } catch (_err) {
        return;
      }

      if (
        url.pathname !== window.location.pathname ||
        url.search !== window.location.search ||
        !url.hash
      ) {
        return;
      }

      const id = decodeHashId(url.hash.slice(1));
      const target = document.getElementById(id);
      revealProofForTarget(target);
    });
  }

  function bootstrap() {
    initAllProofs();
    revealProofForHashTarget();
    bindHashAwareLinkExpansion();
    window.addEventListener("hashchange", revealProofForHashTarget);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
