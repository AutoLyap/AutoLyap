(() => {
  const COPY_BUTTON_CLASS = "copy-code-button";
  const WRAPPER_CLASS = "copyable-code-block";

  function setCopiedState(button, text) {
    button.textContent = text;
    button.classList.add("copied");
    window.setTimeout(() => {
      button.textContent = "Copy";
      button.classList.remove("copied");
    }, 1400);
  }

  function fallbackCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.left = "-1000px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  }

  async function copyText(text, button) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        setCopiedState(button, "Copied");
        return;
      }
      const ok = fallbackCopy(text);
      setCopiedState(button, ok ? "Copied" : "Failed");
    } catch (_err) {
      const ok = fallbackCopy(text);
      setCopiedState(button, ok ? "Copied" : "Failed");
    }
  }

  function ensureWrapper(pre) {
    if (pre.parentElement && pre.parentElement.classList.contains(WRAPPER_CLASS)) {
      return pre.parentElement;
    }
    const wrapper = document.createElement("div");
    wrapper.className = WRAPPER_CLASS;
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);
    return wrapper;
  }

  function attachCopyButtons() {
    const blocks = document.querySelectorAll(
      ".rst-content div.highlight pre, .rst-content pre.literal-block"
    );
    blocks.forEach((pre) => {
      if (!(pre instanceof HTMLElement) || pre.dataset.copyButtonBound === "1") {
        return;
      }
      pre.dataset.copyButtonBound = "1";

      const wrapper = ensureWrapper(pre);
      if (wrapper.querySelector(`:scope > .${COPY_BUTTON_CLASS}`)) {
        return;
      }

      const button = document.createElement("button");
      button.type = "button";
      button.className = COPY_BUTTON_CLASS;
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy code to clipboard");
      button.addEventListener("click", () => {
        const text = pre.innerText.replace(/\u00a0/g, " ");
        copyText(text, button);
      });

      wrapper.appendChild(button);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachCopyButtons);
  } else {
    attachCopyButtons();
  }
})();
