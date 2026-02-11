(() => {
  function optimizeContentImages() {
    const images = document.querySelectorAll(".rst-content img");
    images.forEach((img, index) => {
      if (!(img instanceof HTMLImageElement)) {
        return;
      }

      // Keep the first content image eager to protect LCP.
      if (!img.hasAttribute("loading")) {
        img.setAttribute("loading", index === 0 ? "eager" : "lazy");
      }
      if (!img.hasAttribute("decoding")) {
        img.setAttribute("decoding", "async");
      }
      if (!img.hasAttribute("fetchpriority")) {
        img.setAttribute("fetchpriority", index === 0 ? "high" : "low");
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", optimizeContentImages);
  } else {
    optimizeContentImages();
  }
})();
