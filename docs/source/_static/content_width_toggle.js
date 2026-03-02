(() => {
  const BUTTON_ID = "autolyap-width-toggle";
  const STORAGE_KEY = "autolyap:content-width";
  const FULL_WIDTH_CLASS = "autolyap-full-width";

  function getStoredMode() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  }

  function storeMode(isFullWidth) {
    try {
      window.localStorage.setItem(STORAGE_KEY, isFullWidth ? "full" : "normal");
    } catch {
      // Ignore storage failures (privacy mode, blocked storage, etc.).
    }
  }

  function setButtonState(button, isFullWidth) {
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }
    button.textContent = isFullWidth ? "Normal width" : "Full width";
    button.setAttribute("aria-pressed", String(isFullWidth));
    button.title = isFullWidth
      ? "Switch back to the default reading width"
      : "Expand content to fill the window width";
  }

  function applyMode(isFullWidth) {
    if (!(document.body instanceof HTMLBodyElement)) {
      return;
    }
    document.body.classList.toggle(FULL_WIDTH_CLASS, isFullWidth);
    const button = document.getElementById(BUTTON_ID);
    setButtonState(button, isFullWidth);
  }

  function currentModeIsFullWidth() {
    return document.body instanceof HTMLBodyElement
      && document.body.classList.contains(FULL_WIDTH_CLASS);
  }

  function init() {
    if (!(document.body instanceof HTMLBodyElement)) {
      return;
    }

    const isFullWidth = getStoredMode() === "full";
    applyMode(isFullWidth);

    const button = document.getElementById(BUTTON_ID);
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }

    button.addEventListener("click", () => {
      const nextMode = !currentModeIsFullWidth();
      applyMode(nextMode);
      storeMode(nextMode);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
    return;
  }
  init();
})();
